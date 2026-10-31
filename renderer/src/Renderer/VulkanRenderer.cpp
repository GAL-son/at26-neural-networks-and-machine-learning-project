#include "VulkanRenderer.hpp"
#include <stdexcept>
#include <iostream>

namespace Engine::Renderer
{
    VulkanRenderer::VulkanRenderer(VulkanContext &vulkanContext, const Window &window)
        : m_vulkanContext(vulkanContext),
          m_window(window)
    {
    }

    VulkanRenderer::~VulkanRenderer()
    {
        cleanup();
    }

    void VulkanRenderer::init()
    {
        createCommandPool();
        createCommandBuffers();
        createSampler();
        createSyncPrimitives();

        VkExtent2D extent = m_vulkanContext.getSwapchainExtent();

        // 1. Tworzenie i inicjalizacja GeometryPass
        m_geometryPass = std::make_unique<GeometryPass>(m_vulkanContext, extent);
        m_geometryPass->Create();

        // 2. Pobieramy widoki kolorów i doklejamy na koniec widok głębi
        std::vector<VkImageView> compositeInputViews = m_geometryPass->GetTargetImageViews(); // 3 bufory kolorów
        compositeInputViews.push_back(m_geometryPass->GetDepthImageView());                   // Czwarty bufor = Głębokość

        // 3. Tworzenie i inicjalizacja CompositePass z nową listą widoków
        m_compositePass = std::make_unique<CompositePass>(
            m_vulkanContext,
            extent,
            compositeInputViews, // <-- Przekazujemy 3 kolory + 1 głębię
            m_defaultSampler);
        m_compositePass->Create();

        // 4. Inicjalizacja UI
        m_uiManager.init(m_window, m_vulkanContext, m_imguiPool);
    }

    void VulkanRenderer::loadScene(const Engine::Scene::Scene &cpuScene)
    {
        // Upewniamy się, że GPU nie wykonuje w tym momencie prac renderowania
        vkDeviceWaitIdle(m_vulkanContext.getDevice());

        // Tworzymy nową scenę GPU przy użyciu surowej sceny CPU i wspólnego Command Poola
        m_activeGpuScene = std::make_unique<Renderer::Scene>(m_vulkanContext, m_commandPool, cpuScene);
    }

    void VulkanRenderer::cleanup()
    {
        VkDevice device = m_vulkanContext.getDevice();
        if (device == VK_NULL_HANDLE)
            return;

        vkDeviceWaitIdle(device);

        // 1. Najpierw niszczymy obiekty alokujące pamięć przez VMA (scena, passy)
        m_activeGpuScene.reset();

        if (m_compositePass)
        {
            m_compositePass->Cleanup();
            m_compositePass.reset();
        }

        if (m_geometryPass)
        {
            m_geometryPass->Cleanup();
            m_geometryPass.reset();
        }

        m_uiManager.shutdown(m_vulkanContext, m_imguiPool);

        // 2. Dopiero na końcu reszta zasobów logicznych Vulcana (samplery, semafory itp.)
        if (m_defaultSampler)
            vkDestroySampler(device, m_defaultSampler, nullptr);

        if (m_imageAvailableSemaphore)
            vkDestroySemaphore(device, m_imageAvailableSemaphore, nullptr);

        for (auto sem : m_renderFinishedSemaphores)
        {
            if (sem)
                vkDestroySemaphore(device, sem, nullptr);
        }
        m_renderFinishedSemaphores.clear();

        if (m_inFlightFence)
            vkDestroyFence(device, m_inFlightFence, nullptr);

        if (m_commandPool)
            vkDestroyCommandPool(device, m_commandPool, nullptr);
    }

    void VulkanRenderer::renderFrame(Engine::Camera &camera, uint32_t processedFrames, float frameTimeMs, float passTimeMs)
    {
        // 1. Pacing & Acquisition
        waitForInFlightFence();
        uint32_t imageIndex = acquireNextSwapchainImage();

        // 2. Rejestrowanie komend renderowania
        vkResetCommandBuffer(m_commandBuffer, 0);
        beginCommandBuffer();

        VkExtent2D extent = m_vulkanContext.getSwapchainExtent();

        // A. Pass Geometryczny (MRT -> 4 bufory wyjściowe)
        m_geometryPass->Begin(m_commandBuffer, extent);
        if (m_activeGpuScene)
        {
            m_geometryPass->Render(m_commandBuffer, *m_activeGpuScene, camera);
        }
        m_geometryPass->End(m_commandBuffer);

        // B. Przygotowanie swapchaina pod rendering kompozycji oraz UI
        prepareSwapchainForRendering(imageIndex);

        // C. Composite Pass & UI Overlay na bufor Swapchain
        m_uiManager.beginFrame();
        m_uiManager.renderStatsAndGrid(processedFrames, frameTimeMs, passTimeMs);

        VkRenderingAttachmentInfo colorAttachmentInfo{};
        colorAttachmentInfo.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        colorAttachmentInfo.imageView = m_vulkanContext.getSwapchainImageViews()[imageIndex];
        colorAttachmentInfo.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachmentInfo.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachmentInfo.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachmentInfo.clearValue.color = {{0.0f, 0.0f, 0.0f, 1.0f}};

        VkRenderingInfo renderingInfo{};
        renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        renderingInfo.renderArea.offset = {0, 0};
        renderingInfo.renderArea.extent = extent;
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachments = &colorAttachmentInfo;

        vkCmdBeginRendering(m_commandBuffer, &renderingInfo);

        VkViewport viewport{0.0f, 0.0f, (float)extent.width, (float)extent.height, 0.0f, 1.0f};
        vkCmdSetViewport(m_commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{{0, 0}, extent};
        vkCmdSetScissor(m_commandBuffer, 0, 1, &scissor);

        // Rysujemy siatkę 2x2 z tekstur MRT za pomocą CompositePass
        if (m_activeGpuScene)
        {
            m_compositePass->Render(m_commandBuffer, *m_activeGpuScene, camera);
        }

        // Rysujemy ImGui w tym samym subpassie/rendering passie
        m_uiManager.renderDrawData(m_commandBuffer);

        vkCmdEndRendering(m_commandBuffer);

        // D. Przygotowanie obrazu do zaprezentowania na ekranie
        prepareSwapchainForPresentation(imageIndex);

        vkEndCommandBuffer(m_commandBuffer);

        // 3. Wysyłanie do GPU & Present
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

    void VulkanRenderer::createSampler()
    {
        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

        if (vkCreateSampler(m_vulkanContext.getDevice(), &samplerInfo, nullptr, &m_defaultSampler) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create default sampler!");
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