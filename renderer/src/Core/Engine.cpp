#include "Engine.hpp"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <iostream>
#include <thread>

Engine::Engine(bool headless) : m_headless(headless), m_window(1280, 720, "AI Antialiaser Renderer", headless), m_vulkanContext(m_window)
{
    if (!m_headless)
    {
        initImGui();
    }
}

Engine::~Engine()
{
    // Cleanup ImGui / Vulkan
    if (!m_headless)
    {
        vkDeviceWaitIdle(m_vulkanContext.getDevice());
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        if (m_imguiPool)
            vkDestroyDescriptorPool(m_vulkanContext.getDevice(), m_imguiPool, nullptr);
        if (m_commandPool)
            vkDestroyCommandPool(m_vulkanContext.getDevice(), m_commandPool, nullptr);
    }
}

void Engine::initImGui() {
    // 1. Create Descriptor Pool for ImGui
    VkDescriptorPoolSize pool_sizes[] = {
        { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
    };
    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = 1000 * IM_ARRAYSIZE(pool_sizes);
    pool_info.poolSizeCount = (uint32_t)IM_ARRAYSIZE(pool_sizes);
    pool_info.pPoolSizes = pool_sizes;
    vkCreateDescriptorPool(m_vulkanContext.getDevice(), &pool_info, nullptr, &m_imguiPool);

    // 2. Initialize ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();

    // 3. Initialize GLFW binding
    ImGui_ImplGlfw_InitForVulkan(m_window.getGLFWWindow(), true);

    // 4. Initialize Vulkan binding with Dynamic Rendering
    VkFormat swapchainFormat = m_vulkanContext.getSwapchainImageFormat();
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = m_vulkanContext.getInstance();
    init_info.PhysicalDevice = m_vulkanContext.getPhysicalDevice();
    init_info.Device = m_vulkanContext.getDevice();
    init_info.QueueFamily = m_vulkanContext.getGraphicsQueueFamily();
    init_info.Queue = m_vulkanContext.getGraphicsQueue();
    init_info.DescriptorPool = m_imguiPool;
    init_info.MinImageCount = 2;
    init_info.ImageCount = 2;
    
    // Enable dynamic rendering and set format inside PipelineInfoMain
    init_info.UseDynamicRendering = true;
    init_info.PipelineInfoMain.PipelineRenderingCreateInfo = {};
    init_info.PipelineInfoMain.PipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    init_info.PipelineInfoMain.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    init_info.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats = &swapchainFormat;

    ImGui_ImplVulkan_Init(&init_info);
    
    // Note: Fonts are now uploaded automatically on the first call to ImGui_ImplVulkan_NewFrame()!
}

void Engine::run() {
    std::cout << "Running loop in " << (m_headless ? "HEADLESS" : "WINDOWED") << " mode...\n";
    m_lastTime = std::chrono::high_resolution_clock::now();

    if (m_headless) {
        while (!m_window.shouldClose()) {
            auto currentTime = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float, std::milli> delta = currentTime - m_lastTime;
            m_lastTime = currentTime;

            m_frameTimeMs = delta.count();
            m_passTimeMs = m_frameTimeMs * 0.75f;
            m_processedFrames++;

            m_window.pollEvents();
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
            if (m_processedFrames > 150) break;
        }
        std::cout << "Headless batch generation complete. Exiting.\n";
        return;
    }

    // --- WINDOWED MODE SETUP ---
    VkDevice device = m_vulkanContext.getDevice();
    VkQueue graphicsQueue = m_vulkanContext.getGraphicsQueue();

    // 1. Create a Command Pool
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = m_vulkanContext.getGraphicsQueueFamily();
    if (vkCreateCommandPool(device, &poolInfo, nullptr, &m_commandPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create command pool!");
    }

    // 2. Allocate a Command Buffer
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = m_commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

    // 3. Create Synchronization Objects (Semaphores & Fence)
    VkSemaphore imageAvailableSemaphore, renderFinishedSemaphore;
    VkFence inFlightFence;

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphore);
    vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphore);
    vkCreateFence(device, &fenceInfo, nullptr, &inFlightFence);

    // --- MAIN WINDOWED RENDER LOOP ---
    while (!m_window.shouldClose()) {
        auto currentTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> delta = currentTime - m_lastTime;
        m_lastTime = currentTime;

        m_frameTimeMs = delta.count();
        m_passTimeMs = m_frameTimeMs * 0.75f;
        m_processedFrames++;

        m_window.pollEvents();

        // Wait for previous frame
        vkWaitForFences(device, 1, &inFlightFence, VK_TRUE, UINT64_MAX);
        vkResetFences(device, 1, &inFlightFence);

        // Acquire image from swapchain
        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(device, m_vulkanContext.getSwapchain(), UINT64_MAX, 
                                                imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);
        if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            throw std::runtime_error("Failed to acquire swapchain image!");
        }

        vkResetCommandBuffer(commandBuffer, 0);

        // Begin Recording Command Buffer
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        // Transition swapchain image layout to COLOR_ATTACHMENT_OPTIMAL
        VkImageMemoryBarrier barrierToColor{};
        barrierToColor.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrierToColor.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrierToColor.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrierToColor.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrierToColor.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrierToColor.image = m_vulkanContext.getSwapchainImages()[imageIndex];
        barrierToColor.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrierToColor.subresourceRange.baseMipLevel = 0;
        barrierToColor.subresourceRange.levelCount = 1;
        barrierToColor.subresourceRange.baseArrayLayer = 0;
        barrierToColor.subresourceRange.layerCount = 1;
        barrierToColor.srcAccessMask = 0;
        barrierToColor.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &barrierToColor);

        // --- ImGui Frame Setup ---
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Stats Window
        ImGui::Begin("Renderer Control & Stats");
        ImGui::Text("Mode: Windowed Visualization");
        ImGui::Text("Processed Frames: %u", m_processedFrames);
        ImGui::Text("Frame Time: %.2f ms", m_frameTimeMs);
        ImGui::Text("Pass Time: %.2f ms", m_passTimeMs);
        ImGui::End();

        // 2x2 Buffer View Simulation Window
        ImGui::Begin("Buffers (2x2 Grid View)");
        ImGui::Columns(2, "BufferColumns", false);
        ImGui::Text("1. G-Buffer (Aliased)");
        ImGui::Button("Color A", ImVec2(250, 150));
        ImGui::NextColumn();
        
        ImGui::Text("2. Depth & Normals");
        ImGui::Button("Color B", ImVec2(250, 150));
        ImGui::NextColumn();
        
        ImGui::Text("3. Motion Vectors");
        ImGui::Button("Color C", ImVec2(250, 150));
        ImGui::NextColumn();
        
        ImGui::Text("4. SSAAx32 Ground Truth");
        ImGui::Button("Color D", ImVec2(250, 150));
        ImGui::Columns(1);
        ImGui::End();

        ImGui::Render();

        // --- Dynamic Rendering to Render Swapchain Image ---
        VkRenderingAttachmentInfo colorAttachmentInfo{};
        colorAttachmentInfo.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        colorAttachmentInfo.imageView = m_vulkanContext.getSwapchainImageViews()[imageIndex];
        colorAttachmentInfo.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachmentInfo.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachmentInfo.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachmentInfo.clearValue = {{{0.1f, 0.1f, 0.15f, 1.0f}}}; // Dark blue-gray background

        VkRenderingInfo renderingInfo{};
        renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        renderingInfo.renderArea.extent = m_vulkanContext.getSwapchainExtent();
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachments = &colorAttachmentInfo;

        vkCmdBeginRendering(commandBuffer, &renderingInfo);

        // Render ImGui Draw Data
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);

        vkCmdEndRendering(commandBuffer);

        // Transition swapchain image layout to PRESENT_SRC_KHR
        VkImageMemoryBarrier barrierToPresent{};
        barrierToPresent.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrierToPresent.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrierToPresent.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        barrierToPresent.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrierToPresent.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrierToPresent.image = m_vulkanContext.getSwapchainImages()[imageIndex];
        barrierToPresent.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrierToPresent.subresourceRange.baseMipLevel = 0;
        barrierToPresent.subresourceRange.levelCount = 1;
        barrierToPresent.subresourceRange.baseArrayLayer = 0;
        barrierToPresent.subresourceRange.layerCount = 1;
        barrierToPresent.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        barrierToPresent.dstAccessMask = 0;

        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &barrierToPresent);

        vkEndCommandBuffer(commandBuffer);

        // Submit Command Buffer
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        
        VkSemaphore waitSemaphores[] = {imageAvailableSemaphore};
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        VkSemaphore signalSemaphores[] = {renderFinishedSemaphore};
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFence) != VK_SUCCESS) {
            throw std::runtime_error("Failed to submit draw command buffer!");
        }

        // Present Image
        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;
        
        VkSwapchainKHR swapchains[] = {m_vulkanContext.getSwapchain()};
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapchains;
        presentInfo.pImageIndices = &imageIndex;

        vkQueuePresentKHR(m_vulkanContext.getGraphicsQueue(), &presentInfo);
    }

    vkDeviceWaitIdle(device);

    // Cleanup local sync primitives
    vkDestroySemaphore(device, imageAvailableSemaphore, nullptr);
    vkDestroySemaphore(device, renderFinishedSemaphore, nullptr);
    vkDestroyFence(device, inFlightFence, nullptr);
}