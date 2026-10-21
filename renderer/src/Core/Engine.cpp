#include "Engine.hpp"
#include <iostream>
#include <thread>

Engine::Engine(bool headless)
    : m_headless(headless),
      m_window(1280, 720, "AI Antialiaser Renderer", headless),
      m_vulkanContext(m_window),
      m_compositePass(m_vulkanContext)
{
    if (!m_headless)
    {
        m_uiManager.init(m_window, m_vulkanContext, m_imguiPool);
        m_compositePass.init();
    }
}

Engine::~Engine()
{
    if (!m_headless)
    {
        VkDevice device = m_vulkanContext.getDevice();
        vkDeviceWaitIdle(device);

        m_compositePass.cleanup();
        m_uiManager.shutdown(m_vulkanContext, m_imguiPool);
        if (m_commandPool)
            vkDestroyCommandPool(device, m_commandPool, nullptr);
    }
}

void Engine::run()
{
    std::cout << "Running loop in " << (m_headless ? "HEADLESS" : "WINDOWED") << " mode...\n";
    m_lastTime = std::chrono::high_resolution_clock::now();

    if (m_headless)
    {
        runHeadless();
    }
    else
    {
        runWindowed();
    }
}

void Engine::runHeadless()
{
    while (!m_window.shouldClose())
    {
        auto currentTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> delta = currentTime - m_lastTime;
        m_lastTime = currentTime;

        m_frameTimeMs = delta.count();
        m_passTimeMs = m_frameTimeMs * 0.75f;
        m_processedFrames++;

        m_window.pollEvents();
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
        if (m_processedFrames > 150)
            break;
    }
    std::cout << "Headless batch generation complete. Exiting.\n";
}

void Engine::runWindowed()
{
    VkDevice device = m_vulkanContext.getDevice();
    VkQueue graphicsQueue = m_vulkanContext.getGraphicsQueue();

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = m_vulkanContext.getGraphicsQueueFamily();
    if (vkCreateCommandPool(device, &poolInfo, nullptr, &m_commandPool) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create command pool!");
    }

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = m_commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

    VkSemaphore imageAvailableSemaphore;
    auto swapchainImages = m_vulkanContext.getSwapchainImages();
    size_t imageCount = swapchainImages.size();

    std::vector<VkSemaphore> renderFinishedSemaphores(imageCount);
    VkFence inFlightFence;

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphore);
    for (size_t i = 0; i < imageCount; i++)
    {
        vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]);
    }
    vkCreateFence(device, &fenceInfo, nullptr, &inFlightFence);

    while (!m_window.shouldClose())
    {
        auto currentTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> delta = currentTime - m_lastTime;
        m_lastTime = currentTime;

        m_frameTimeMs = delta.count();
        m_passTimeMs = m_frameTimeMs * 0.75f;
        m_processedFrames++;

        m_window.pollEvents();

        vkWaitForFences(device, 1, &inFlightFence, VK_TRUE, UINT64_MAX);
        vkResetFences(device, 1, &inFlightFence);

        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(device, m_vulkanContext.getSwapchain(), UINT64_MAX,
                                                imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);
        if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
        {
            throw std::runtime_error("Failed to acquire swapchain image!");
        }

        vkResetCommandBuffer(commandBuffer, 0);

        // --- COMMAND BUFFER RECORDING ---
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        // 1. Clear offscreen buffers
        m_compositePass.recordBufferClears(commandBuffer);

        // 2. Transition swapchain image to color attachment
        VkImageMemoryBarrier barrierToColor{};
        barrierToColor.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrierToColor.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrierToColor.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrierToColor.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrierToColor.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrierToColor.image = m_vulkanContext.getSwapchainImages()[imageIndex];
        barrierToColor.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barrierToColor.srcAccessMask = 0;
        barrierToColor.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &barrierToColor);

        // 3. UI Frame Setup
        m_uiManager.beginFrame();
        m_uiManager.renderStatsAndGrid(m_processedFrames, m_frameTimeMs, m_passTimeMs);

        // 4. Dynamic Rendering Block
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

        vkCmdBeginRendering(commandBuffer, &renderingInfo);

        VkViewport viewport{0.0f, 0.0f, (float)m_vulkanContext.getSwapchainExtent().width, (float)m_vulkanContext.getSwapchainExtent().height, 0.0f, 1.0f};
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{{0, 0}, m_vulkanContext.getSwapchainExtent()};
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        m_compositePass.drawComposite(commandBuffer);
        m_uiManager.renderDrawData(commandBuffer);

        vkCmdEndRendering(commandBuffer);

        // 5. Transition swapchain to present
        VkImageMemoryBarrier barrierToPresent{};
        barrierToPresent.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrierToPresent.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrierToPresent.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        barrierToPresent.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrierToPresent.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrierToPresent.image = m_vulkanContext.getSwapchainImages()[imageIndex];
        barrierToPresent.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barrierToPresent.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        barrierToPresent.dstAccessMask = 0;

        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &barrierToPresent);

        vkEndCommandBuffer(commandBuffer);

        // --- Submit & Present ---
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkSemaphore waitSemaphores[] = {imageAvailableSemaphore};
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[imageIndex]};
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFence) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to submit draw command buffer!");
        }

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;

        VkSwapchainKHR swapchains[] = {m_vulkanContext.getSwapchain()};
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapchains;
        presentInfo.pImageIndices = &imageIndex;

        vkQueuePresentKHR(graphicsQueue, &presentInfo);
    }

    vkDeviceWaitIdle(device);

    vkDestroySemaphore(device, imageAvailableSemaphore, nullptr);
    for (auto sem : renderFinishedSemaphores)
    {
        vkDestroySemaphore(device, sem, nullptr);
    }
    vkDestroyFence(device, inFlightFence, nullptr);
}