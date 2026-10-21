#include "Engine.hpp"
#include <iostream>
#include <thread>

#include "../Utils/FileUtils.hpp"

Engine::Engine(bool headless) : m_headless(headless), m_window(1280, 720, "AI Antialiaser Renderer", headless), m_vulkanContext(m_window)
{
    if (!m_headless)
    {
        m_uiManager.init(m_window, m_vulkanContext, m_imguiPool);

        initOffscreenBuffers();
        initCompositePipeline();
        std::cout << "DEBUG END" << std::endl;
    }
}

Engine::~Engine()
{
    if (!m_headless)
    {
        VkDevice device = m_vulkanContext.getDevice();
        vkDeviceWaitIdle(device);

        cleanupOffscreenBuffers(); // <-- Clean up VMA images here!

        m_uiManager.shutdown(m_vulkanContext, m_imguiPool);
        if (m_commandPool)
            vkDestroyCommandPool(device, m_commandPool, nullptr);
    }
}

void Engine::initCompositePipeline()
{
    std::cout << "DEBUG 1" << std::endl;
    VkDevice device = m_vulkanContext.getDevice();

    std::cout << "DEBUG 2" << std::endl;

    // 1. Create Descriptor Set Layout for 4 combined image samplers
    VkDescriptorSetLayoutBinding bindings[4];
    for (int i = 0; i < 4; i++)
    {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings[i].pImmutableSamplers = nullptr;
    }

    std::cout << "DEBUG 4" << std::endl;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 4;
    layoutInfo.pBindings = bindings;

    std::cout << "DEBUG 5" << std::endl;

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &m_compositeDescriptorLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create composite descriptor set layout!");
    }

    std::cout << "DEBUG 6" << std::endl;

    // 2. Allocate Descriptor Set from m_imguiPool (or a separate pool, but m_imguiPool works if sized large enough)
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSize.descriptorCount = 4;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = 1;

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &m_compositePool) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create composite descriptor pool!");
    }

    // 2. Allocate Descriptor Set from our dedicated composite pool
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = m_compositePool; // <-- Use dedicated pool here!
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &m_compositeDescriptorLayout;

    std::cout << "DEBUG 7" << std::endl;

    if (vkAllocateDescriptorSets(device, &allocInfo, &m_compositeDescriptorSet) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to allocate composite descriptor set!");
    }

    std::cout << "DEBUG 8" << std::endl;

    // Update descriptor set to point to our 4 offscreen buffer image views & samplers
    VkDescriptorImageInfo imageInfos[4];
    VkWriteDescriptorSet descriptorWrites[4] = {};

    std::cout << "DEBUG 9" << std::endl;

    for (int i = 0; i < 4; i++)
    {
        imageInfos[i].sampler = m_buffers[i].sampler;
        imageInfos[i].imageView = m_buffers[i].imageView;
        imageInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[i].dstSet = m_compositeDescriptorSet;
        descriptorWrites[i].dstBinding = i;
        descriptorWrites[i].dstArrayElement = 0;
        descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[i].descriptorCount = 1;
        descriptorWrites[i].pImageInfo = &imageInfos[i];
        descriptorWrites[i].pBufferInfo = nullptr;
        descriptorWrites[i].pTexelBufferView = nullptr;
    }
    vkUpdateDescriptorSets(device, 4, descriptorWrites, 0, nullptr);

    std::cout << "DEBUG 10" << std::endl;

    // 3. Create Pipeline Layout
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &m_compositeDescriptorLayout;

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &m_compositePipelineLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create composite pipeline layout!");
    }

    std::cout << "DEBUG 11" << std::endl;

    // 4. Create Graphics Pipeline with Dynamic Rendering
    auto vertShaderCode = Utils::readFile("shaders/composite.vert.spv");
    auto fragShaderCode = Utils::readFile("shaders/composite.frag.spv");

    VkShaderModule vertModule = createShaderModule(vertShaderCode);
    VkShaderModule fragModule = createShaderModule(fragShaderCode);

    VkPipelineShaderStageCreateInfo vertStageInfo{};
    vertStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStageInfo.module = vertModule;
    vertStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragStageInfo{};
    fragStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStageInfo.module = fragModule;
    fragStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = {vertStageInfo, fragStageInfo};

    // Fullscreen triangle uses no vertex inputs
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    std::vector<VkDynamicState> dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    VkFormat swapchainFormat = m_vulkanContext.getSwapchainImageFormat();
    VkPipelineRenderingCreateInfo pipelineRenderingCreateInfo{};
    pipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    pipelineRenderingCreateInfo.colorAttachmentCount = 1;
    pipelineRenderingCreateInfo.pColorAttachmentFormats = &swapchainFormat;

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.pNext = &pipelineRenderingCreateInfo;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = m_compositePipelineLayout;
    pipelineInfo.renderPass = VK_NULL_HANDLE; // Using dynamic rendering!

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_compositePipeline) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create composite graphics pipeline!");
    }

    vkDestroyShaderModule(device, vertModule, nullptr);
    vkDestroyShaderModule(device, fragModule, nullptr);
}

VkShaderModule Engine::createShaderModule(const std::vector<char> &code)
{
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(m_vulkanContext.getDevice(), &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create shader module!");
    }
    return shaderModule;
}

void Engine::cleanupOffscreenBuffers()
{
    VkDevice device = m_vulkanContext.getDevice();
    VmaAllocator allocator = m_vulkanContext.getAllocator();

    for (int i = 0; i < 4; i++)
    {
        if (m_buffers[i].imageView != VK_NULL_HANDLE)
        {
            vkDestroyImageView(device, m_buffers[i].imageView, nullptr);
            m_buffers[i].imageView = VK_NULL_HANDLE;
        }
        if (m_buffers[i].sampler != VK_NULL_HANDLE)
        {
            vkDestroySampler(device, m_buffers[i].sampler, nullptr);
            m_buffers[i].sampler = VK_NULL_HANDLE;
        }
        if (m_buffers[i].image != VK_NULL_HANDLE && m_buffers[i].allocation != VK_NULL_HANDLE)
        {
            vmaDestroyImage(allocator, m_buffers[i].image, m_buffers[i].allocation);
            m_buffers[i].image = VK_NULL_HANDLE;
            m_buffers[i].allocation = VK_NULL_HANDLE;
        }
    }
}

void Engine::initOffscreenBuffers()
{
    VkDevice device = m_vulkanContext.getDevice();
    VmaAllocator allocator = m_vulkanContext.getAllocator();
    VkExtent2D extent = m_vulkanContext.getSwapchainExtent();

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = extent.width;
    imageInfo.extent.height = extent.height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;

    for (int i = 0; i < 4; i++)
    {
        vmaCreateImage(allocator, &imageInfo, &allocInfo, &m_buffers[i].image, &m_buffers[i].allocation, nullptr);

        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = m_buffers[i].image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;
        vkCreateImageView(device, &viewInfo, nullptr, &m_buffers[i].imageView);

        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        vkCreateSampler(device, &samplerInfo, nullptr, &m_buffers[i].sampler);
    }
}

void Engine::run()
{
    std::cout << "Running loop in " << (m_headless ? "HEADLESS" : "WINDOWED") << " mode...\n";
    m_lastTime = std::chrono::high_resolution_clock::now();

    if (m_headless)
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
        return;
    }

    // --- WINDOWED MODE SETUP ---
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

        // 1. BEGIN COMMAND BUFFER RECORDING (Must be first!)
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        VkClearColorValue clearColors[4] = {
            {{1.0f, 0.0f, 0.0f, 1.0f}}, // Red (G-Buffer)
            {{0.0f, 1.0f, 0.0f, 1.0f}}, // Green (Depth/Normals)
            {{0.0f, 0.0f, 1.0f, 1.0f}}, // Blue (Motion Vectors)
            {{1.0f, 1.0f, 0.0f, 1.0f}}  // Yellow (SSAA Ground Truth)
        };

        for (int i = 0; i < 4; i++)
        {
            // Barrier: UNDEFINED -> TRANSFER_DST
            VkImageMemoryBarrier barrierToDst{};
            barrierToDst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrierToDst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrierToDst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrierToDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrierToDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrierToDst.image = m_buffers[i].image;
            barrierToDst.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            barrierToDst.subresourceRange.baseMipLevel = 0;
            barrierToDst.subresourceRange.levelCount = 1;
            barrierToDst.subresourceRange.baseArrayLayer = 0;
            barrierToDst.subresourceRange.layerCount = 1;
            barrierToDst.srcAccessMask = 0;
            barrierToDst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &barrierToDst);

            VkImageSubresourceRange range{};
            range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            range.baseMipLevel = 0;
            range.levelCount = 1;
            range.baseArrayLayer = 0;
            range.layerCount = 1;

            vkCmdClearColorImage(commandBuffer, m_buffers[i].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearColors[i], 1, &range);

            // Barrier: TRANSFER_DST -> SHADER_READ_ONLY_OPTIMAL
            VkImageMemoryBarrier barrierToShader{};
            barrierToShader.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrierToShader.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrierToShader.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrierToShader.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrierToShader.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrierToShader.image = m_buffers[i].image;
            barrierToShader.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            barrierToShader.subresourceRange.baseMipLevel = 0;
            barrierToShader.subresourceRange.levelCount = 1;
            barrierToShader.subresourceRange.baseArrayLayer = 0;
            barrierToShader.subresourceRange.layerCount = 1;
            barrierToShader.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrierToShader.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &barrierToShader);
        }

        // --- 2. TRANSITION SWAPCHAIN IMAGE TO COLOR_ATTACHMENT ---
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

        // --- 3. UI FRAME SETUP ---
        m_uiManager.beginFrame();
        m_uiManager.renderStatsAndGrid(m_processedFrames, m_frameTimeMs, m_passTimeMs);

        // --- 4. DYNAMIC RENDERING (BACKGROUND COMPOSITE + IMGUI) ---
        VkRenderingAttachmentInfo colorAttachmentInfo{};
        colorAttachmentInfo.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        colorAttachmentInfo.imageView = m_vulkanContext.getSwapchainImageViews()[imageIndex];
        colorAttachmentInfo.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachmentInfo.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachmentInfo.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        VkRenderingInfo renderingInfo{};
        renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        renderingInfo.renderArea.extent = m_vulkanContext.getSwapchainExtent();
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachments = &colorAttachmentInfo;

        vkCmdBeginRendering(commandBuffer, &renderingInfo);

        // Set Dynamic Viewport & Scissor
        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(m_vulkanContext.getSwapchainExtent().width);
        viewport.height = static_cast<float>(m_vulkanContext.getSwapchainExtent().height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = m_vulkanContext.getSwapchainExtent();
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        // Draw the 2x2 Tiled Background via Composite Pipeline
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_compositePipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_compositePipelineLayout, 0, 1, &m_compositeDescriptorSet, 0, nullptr);
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);

        // Render ImGui UI on top of the background
        m_uiManager.renderDrawData(commandBuffer);

        vkCmdEndRendering(commandBuffer);

        // --- 5. TRANSITION SWAPCHAIN TO PRESENT ---
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
    vkDestroySemaphore(device, renderFinishedSemaphore, nullptr);
    vkDestroyFence(device, inFlightFence, nullptr);
}