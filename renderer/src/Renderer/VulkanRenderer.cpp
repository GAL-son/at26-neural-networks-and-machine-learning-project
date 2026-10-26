#include "VulkanRenderer.hpp"
#include <stdexcept>
#include <iostream>

namespace Engine::Renderer
{
    VulkanRenderer::VulkanRenderer(VulkanContext &vulkanContext, const Window &window)
        : m_vulkanContext(vulkanContext),
          m_window(window),
          m_compositePass(vulkanContext),
          m_geometryPass(vulkanContext)
    {
    }

    VulkanRenderer::~VulkanRenderer()
    {
        cleanup();
    }

    void VulkanRenderer::init()
    {
        m_geometryPass.init();
        m_compositePass.init();
        m_uiManager.init(m_window, m_vulkanContext, m_imguiPool);

        createCommandPool();
        createCommandBuffers();
        createSyncPrimitives();
    }

    void VulkanRenderer::cleanup()
    {
        VkDevice device = m_vulkanContext.getDevice();
        if (device == VK_NULL_HANDLE)
            return;

        vkDeviceWaitIdle(device);

        m_compositePass.cleanup();
        m_uiManager.shutdown(m_vulkanContext, m_imguiPool);

        if (m_imageAvailableSemaphore)
            vkDestroySemaphore(device, m_imageAvailableSemaphore, nullptr);

        for (auto sem : m_renderFinishedSemaphores)
        {
            if (sem)
                vkDestroySemaphore(device, sem, nullptr);
        }
        if (m_inFlightFence)
            vkDestroyFence(device, m_inFlightFence, nullptr);
        if (m_commandPool)
            vkDestroyCommandPool(device, m_commandPool, nullptr);
    }

    void VulkanRenderer::renderFrame(Scene::Scene *scene, uint32_t processedFrames, float frameTimeMs, float passTimeMs)
    {
        VkDevice device = m_vulkanContext.getDevice();

        // 1. Frame pacing & acquisition
        waitForInFlightFence();
        uint32_t imageIndex = acquireNextSwapchainImage();

        // 2. Command recording loop
        vkResetCommandBuffer(m_commandBuffer, 0);
        beginCommandBuffer();

        recordGeometryAndClears(scene);
        prepareSwapchainForRendering(imageIndex);
        recordUIAndComposition(imageIndex, processedFrames, frameTimeMs, passTimeMs);
        prepareSwapchainForPresentation(imageIndex);

        vkEndCommandBuffer(m_commandBuffer);

        // 3. GPU submission & presentation
        submitCommandBuffer(imageIndex);
        presentSwapchainImage(imageIndex);
    }

    void VulkanRenderer::createCommandPool()
    {
        VkDevice device = m_vulkanContext.getDevice();
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = m_vulkanContext.getGraphicsQueueFamily();

        if (vkCreateCommandPool(device, &poolInfo, nullptr, &m_commandPool) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create command pool!");
        }
    }

    void VulkanRenderer::createCommandBuffers()
    {
        VkDevice device = m_vulkanContext.getDevice();
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = m_commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        if (vkAllocateCommandBuffers(device, &allocInfo, &m_commandBuffer) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to allocate command buffer!");
        }
    }

    void VulkanRenderer::createSyncPrimitives()
    {
        VkDevice device = m_vulkanContext.getDevice();
        VkSemaphoreCreateInfo semaphoreInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        vkCreateSemaphore(device, &semaphoreInfo, nullptr, &m_imageAvailableSemaphore);

        auto swapchainImages = m_vulkanContext.getSwapchainImages();
        m_renderFinishedSemaphores.resize(swapchainImages.size());
        for (size_t i = 0; i < swapchainImages.size(); i++)
        {
            vkCreateSemaphore(device, &semaphoreInfo, nullptr, &m_renderFinishedSemaphores[i]);
        }
        vkCreateFence(device, &fenceInfo, nullptr, &m_inFlightFence);
    }

    void VulkanRenderer::waitForInFlightFence()
    {
        VkDevice device = m_vulkanContext.getDevice();
        vkWaitForFences(device, 1, &m_inFlightFence, VK_TRUE, UINT64_MAX);
        vkResetFences(device, 1, &m_inFlightFence);
    }

    uint32_t VulkanRenderer::acquireNextSwapchainImage()
    {
        VkDevice device = m_vulkanContext.getDevice();
        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(
            device, m_vulkanContext.getSwapchain(), UINT64_MAX,
            m_imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);

        if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
        {
            throw std::runtime_error("Failed to acquire swapchain image!");
        }
        return imageIndex;
    }

    void VulkanRenderer::beginCommandBuffer()
    {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(m_commandBuffer, &beginInfo);
    }

    void VulkanRenderer::recordGeometryAndClears(Scene::Scene *scene)
    {
        m_compositePass.recordBufferClears(m_commandBuffer);

        if (!scene)
            return;

        m_geometryPass.render(
            m_commandBuffer, *scene,
            m_compositePass.getBuffer(0).imageView,
            m_compositePass.getBuffer(0).image,
            m_vulkanContext.getSwapchainExtent());

        m_geometryPass.renderNormals(
            m_commandBuffer, *scene,
            m_compositePass.getBuffer(1).imageView,
            m_compositePass.getBuffer(1).image,
            m_vulkanContext.getSwapchainExtent());
    }

    void VulkanRenderer::prepareSwapchainForRendering(uint32_t imageIndex)
    {
        insertImageMemoryBarrier(
            m_vulkanContext.getSwapchainImages()[imageIndex],
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            0,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
    }

    void VulkanRenderer::prepareSwapchainForPresentation(uint32_t imageIndex)
    {
        insertImageMemoryBarrier(
            m_vulkanContext.getSwapchainImages()[imageIndex],
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            0,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    }

    void VulkanRenderer::insertImageMemoryBarrier(
        VkImage image,
        VkImageLayout oldLayout,
        VkImageLayout newLayout,
        VkAccessFlags srcAccess,
        VkAccessFlags dstAccess,
        VkPipelineStageFlags srcStage,
        VkPipelineStageFlags dstStage)
    {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barrier.srcAccessMask = srcAccess;
        barrier.dstAccessMask = dstAccess;

        vkCmdPipelineBarrier(m_commandBuffer, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    }

    void VulkanRenderer::recordUIAndComposition(uint32_t imageIndex, uint32_t processedFrames, float frameTimeMs, float passTimeMs)
    {
        m_uiManager.beginFrame();
        m_uiManager.renderStatsAndGrid(processedFrames, frameTimeMs, passTimeMs);

        VkRenderingAttachmentInfo colorAttachmentInfo{};
        colorAttachmentInfo.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        colorAttachmentInfo.imageView = m_vulkanContext.getSwapchainImageViews()[imageIndex];
        colorAttachmentInfo.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachmentInfo.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachmentInfo.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        VkRenderingInfo renderingInfo{};
        renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        renderingInfo.renderArea.offset = {0, 0};
        renderingInfo.renderArea.extent = m_vulkanContext.getSwapchainExtent();
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachments = &colorAttachmentInfo;

        vkCmdBeginRendering(m_commandBuffer, &renderingInfo);

        VkViewport viewport{0.0f, 0.0f, (float)m_vulkanContext.getSwapchainExtent().width, (float)m_vulkanContext.getSwapchainExtent().height, 0.0f, 1.0f};
        vkCmdSetViewport(m_commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{{0, 0}, m_vulkanContext.getSwapchainExtent()};
        vkCmdSetScissor(m_commandBuffer, 0, 1, &scissor);

        m_compositePass.drawComposite(m_commandBuffer);
        m_uiManager.renderDrawData(m_commandBuffer);

        vkCmdEndRendering(m_commandBuffer);
    }

    void VulkanRenderer::submitCommandBuffer(uint32_t imageIndex)
    {
        VkQueue graphicsQueue = m_vulkanContext.getGraphicsQueue();

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkSemaphore waitSemaphores[] = {m_imageAvailableSemaphore};
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &m_commandBuffer;

        VkSemaphore signalSemaphores[] = {m_renderFinishedSemaphores[imageIndex]};
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, m_inFlightFence) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to submit draw command buffer!");
        }
    }

    void VulkanRenderer::presentSwapchainImage(uint32_t imageIndex)
    {
        VkQueue graphicsQueue = m_vulkanContext.getGraphicsQueue();

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &m_renderFinishedSemaphores[imageIndex];

        VkSwapchainKHR swapchains[] = {m_vulkanContext.getSwapchain()};
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapchains;
        presentInfo.pImageIndices = &imageIndex;

        vkQueuePresentKHR(graphicsQueue, &presentInfo);
    }
}