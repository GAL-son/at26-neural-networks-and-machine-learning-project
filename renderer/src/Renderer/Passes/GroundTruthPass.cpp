#include "GroundTruthPass.hpp"

#include <array>
#include <stdexcept>
#include <string>

#include "GeometryPass.hpp"


namespace Engine::Renderer
{
    GroundTruthPass::GroundTruthPass(VulkanContext &context, VkExtent2D outputExtent, uint32_t ssaaFactor, VkDescriptorSetLayout materialSetLayout)
        : RenderPass(context),
          m_outputExtent(outputExtent),
          m_ssaaFactor(ssaaFactor),
          m_materialSetLayout(materialSetLayout)
    {
        m_superExtent = VkExtent2D{m_outputExtent.width * m_ssaaFactor, m_outputExtent.height * m_ssaaFactor};
    }

    void GroundTruthPass::Create()
    {
        ValidateLimits();

        m_superColor = CreateImage(
            VK_FORMAT_R8G8B8A8_UNORM,
            m_superExtent,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT);
            
        // Dodano bufor normalnych (analogicznie do GeometryPass)
        m_superNormal = CreateImage(
            VK_FORMAT_R16G16B16A16_SFLOAT,
            m_superExtent,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT);

        m_superDepth = CreateImage(
            VK_FORMAT_D32_SFLOAT,
            m_superExtent,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            VK_IMAGE_ASPECT_DEPTH_BIT);
            
        m_output = CreateImage(
            VK_FORMAT_R16G16B16A16_SFLOAT,
            m_outputExtent,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT);

        CreateRenderPassCustom();
        CreateFramebufferCustom();
        CreateGraphicsPipeline();
        CreateDownsamplePipeline();
    }

    void GroundTruthPass::ValidateLimits() const
    {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(m_context.getPhysicalDevice(), &properties);
        const VkPhysicalDeviceLimits &limits = properties.limits;

        if (m_superExtent.width > limits.maxFramebufferWidth || m_superExtent.height > limits.maxFramebufferHeight ||
            m_superExtent.width > limits.maxImageDimension2D || m_superExtent.height > limits.maxImageDimension2D)
        {
            throw std::runtime_error("GroundTruthPass: Framebuffer size is too large for the device");
        }
    }

    GroundTruthPass::AllocatedImage GroundTruthPass::CreateImage(
        VkFormat format,
        VkExtent2D extent,
        VkImageUsageFlags usage,
        VkImageAspectFlags aspect)
    {
        AllocatedImage result;
        VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.format = format;
        imageInfo.extent = {extent.width, extent.height, 1};
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage = usage;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

        if (vmaCreateImage(m_context.getAllocator(), &imageInfo, &allocInfo, &result.image, &result.allocation, nullptr) != VK_SUCCESS)
        {
            throw std::runtime_error("GroundTruth: Failed to allocate image");
        }

        VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        viewInfo.image = result.image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange = {aspect, 0, 1, 0, 1};

        if (vkCreateImageView(m_device, &viewInfo, nullptr, &result.view) != VK_SUCCESS)
            throw std::runtime_error("GroundTruthPass: Failed to create image view for image:");

        return result;
    }

    void GroundTruthPass::DestroyImage(AllocatedImage &image)
    {
        if (image.view != VK_NULL_HANDLE)
            vkDestroyImageView(m_device, image.view, nullptr);
        if (image.image != VK_NULL_HANDLE)
            vmaDestroyImage(m_context.getAllocator(), image.image, image.allocation);
        image = {};
    }

    void GroundTruthPass::CreateRenderPassCustom()
    {
        // Teraz mamy 2 attachmenty kolorów (Color, Normal) + 1 głębi = 3 attachmenty łącznie
        std::array<VkAttachmentDescription, 3> attachments{};

        // 0: Supersampled color
        attachments[0].format = VK_FORMAT_R8G8B8A8_UNORM;
        attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        // 1: Supersampled normal (dodane)
        attachments[1].format = VK_FORMAT_R16G16B16A16_SFLOAT;
        attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[1].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        // 2: Supersampled depth (przesunięty indeks na 2)
        attachments[2].format = VK_FORMAT_D32_SFLOAT;
        attachments[2].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[2].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[2].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[2].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[2].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[2].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[2].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        std::array<VkAttachmentReference, 2> colorReferences = {
            VkAttachmentReference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
            VkAttachmentReference{1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}
        };
        VkAttachmentReference depthReference{2, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = static_cast<uint32_t>(colorReferences.size());
        subpass.pColorAttachments = colorReferences.data();
        subpass.pDepthStencilAttachment = &depthReference;

        // Zależność dla kolorów / normalnych
        VkSubpassDependency colorDependency{};
        colorDependency.srcSubpass = 0;
        colorDependency.dstSubpass = VK_SUBPASS_EXTERNAL;
        colorDependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        colorDependency.dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        colorDependency.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        colorDependency.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        VkRenderPassCreateInfo renderPassInfo{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &colorDependency;

        if (vkCreateRenderPass(m_device, &renderPassInfo, nullptr, &m_renderPass) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create render pass!");
        }
    }

    void GroundTruthPass::CreateFramebufferCustom()
    {
        // Uwzględnienie widoku normalnych w framebufferze
        std::array<VkImageView, 3> views = {m_superColor.view, m_superNormal.view, m_superDepth.view};

        VkFramebufferCreateInfo framebufferInfo{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        framebufferInfo.renderPass = m_renderPass;
        framebufferInfo.attachmentCount = static_cast<uint32_t>(views.size());
        framebufferInfo.pAttachments = views.data();
        framebufferInfo.width = m_superExtent.width;
        framebufferInfo.height = m_superExtent.height;
        framebufferInfo.layers = 1;

        m_framebuffers.resize(1);
        if (vkCreateFramebuffer(m_device, &framebufferInfo, nullptr, &m_framebuffers[0]) != VK_SUCCESS)
            throw std::runtime_error("Failed to create framebuffer!");
    }

    void GroundTruthPass::CreateGraphicsPipeline()
    {
        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(glm::mat4) * 2;

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &m_materialSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

        if (vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, &m_pipelineLayout) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create geometry pipeline layout!");
        }

        VkShaderModule vertModule = LoadShader("shaders/ground_truth.vert.spv");
        VkShaderModule fragModule = LoadShader("shaders/ground_truth.frag.spv");

        VkPipelineShaderStageCreateInfo shaderStages[] = {
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vertModule, "main", nullptr},
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fragModule, "main", nullptr}};

        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Engine::Scene::Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};
        attributeDescriptions[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Engine::Scene::Vertex, position)};
        attributeDescriptions[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Engine::Scene::Vertex, normal)};
        attributeDescriptions[2] = {2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Engine::Scene::Vertex, texCoord)};

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo viewportState{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rasterizer{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_NONE;
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;

        VkPipelineMultisampleStateCreateInfo multisampling{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo depthStencil{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

        // Dwa attachmenty kolorów wymagają dwóch konfiguracji blendingu
        std::array<VkPipelineColorBlendAttachmentState, 2> colorBlendAttachments{};
        for (auto &attachment : colorBlendAttachments)
        {
            attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            attachment.blendEnable = VK_FALSE;
        }

        VkPipelineColorBlendStateCreateInfo colorBlending{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        colorBlending.attachmentCount = static_cast<uint32_t>(colorBlendAttachments.size());
        colorBlending.pAttachments = colorBlendAttachments.data();

        std::vector<VkDynamicState> dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamicState{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = m_pipelineLayout;
        pipelineInfo.renderPass = m_renderPass;
        pipelineInfo.subpass = 0;

        if (vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pipeline) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create geometry graphics pipeline!");
        }

        vkDestroyShaderModule(m_device, vertModule, nullptr);
        vkDestroyShaderModule(m_device, fragModule, nullptr);
    }

    void GroundTruthPass::CreateDownsamplePipeline()
    {
        std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
        bindings[0] = {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

        VkDescriptorSetLayoutCreateInfo setLayoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        setLayoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        setLayoutInfo.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(m_device, &setLayoutInfo, nullptr, &m_downsampleSetLayout) != VK_SUCCESS)
            throw std::runtime_error("Failed to create descriptor set layout!");

        std::array<VkDescriptorPoolSize, 2> poolSizes = {{
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
        }};

        VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_downsamplePool) != VK_SUCCESS)
            throw std::runtime_error("Failed to create descriptor pool!");

        VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocInfo.descriptorPool = m_downsamplePool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &m_downsampleSetLayout;
        if (vkAllocateDescriptorSets(m_device, &allocInfo, &m_downsampleSet) != VK_SUCCESS)
            throw std::runtime_error("Failed to allocate descriptor sets!");

        VkDescriptorImageInfo superInfo{VK_NULL_HANDLE, m_superColor.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkDescriptorImageInfo outputInfo{VK_NULL_HANDLE, m_output.view, VK_IMAGE_LAYOUT_GENERAL};

        std::array<VkWriteDescriptorSet, 2> writes{};
        writes[0] = {
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            m_downsampleSet,
            0,
            0,
            1,
            VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            &superInfo,
            nullptr,
            nullptr,
        };
        writes[1] = {
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            m_downsampleSet,
            1,
            0,
            1,
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            &outputInfo,
            nullptr,
            nullptr,
        };
        vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

        VkPushConstantRange pushConstantRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int32_t)};
        VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &m_downsampleSetLayout;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pushConstantRange;
        if (vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &m_downsamplePipelineLayout) != VK_SUCCESS)
            throw std::runtime_error("Failed to create pipeline layout!");

        VkShaderModule computeModule = LoadShader("shaders/ssaa_downsample.comp.spv");

        VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipelineInfo.stage = {
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            nullptr,
            0,
            VK_SHADER_STAGE_COMPUTE_BIT,
            computeModule,
            "main",
            nullptr};
        pipelineInfo.layout = m_downsamplePipelineLayout;

        if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_downsamplePipeline) != VK_SUCCESS)
            throw std::runtime_error("Failed to create compute pipeline!");

        vkDestroyShaderModule(m_device, computeModule, nullptr);
    }

    void GroundTruthPass::Begin(VkCommandBuffer commandBuffer, VkExtent2D /*extent*/)
    {
        // 3 wartości czyszczenia: Color, Normal, Depth
        std::array<VkClearValue, 3> clearValues{};
        clearValues[0].color = {{0.1f, 0.1f, 0.1f, 1.0f}};
        clearValues[1].color = {{0.0f, 0.0f, 0.0f, 1.0f}}; // Wyczyszczenie normalnych
        clearValues[2].depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo renderPassInfo{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        renderPassInfo.renderPass = m_renderPass;
        renderPassInfo.framebuffer = m_framebuffers[0];
        renderPassInfo.renderArea = {{0, 0}, m_superExtent};
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport viewport({0.0f, 0.0f, static_cast<float>(m_superExtent.width), static_cast<float>(m_superExtent.height), 0.0f, 1.0f});
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{{0, 0}, m_superExtent};
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
    }

    void GroundTruthPass::Render(VkCommandBuffer commandBuffer, Renderer::Scene &scene, Camera &camera)
    {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);

        glm::mat4 vp = camera.getVPMatrix();
        vkCmdPushConstants(commandBuffer, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(vp), &vp);

        scene.Draw(commandBuffer, m_pipelineLayout);
    }

    void GroundTruthPass::End(VkCommandBuffer commandBuffer)
    {
        vkCmdEndRenderPass(commandBuffer);

        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = m_output.image;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        vkCmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0,
            nullptr,
            0,
            nullptr,
            1,
            &barrier);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_downsamplePipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_downsamplePipelineLayout, 0, 1, &m_downsampleSet, 0, nullptr);

        int32_t factor = static_cast<int32_t>(m_ssaaFactor);
        vkCmdPushConstants(commandBuffer, m_downsamplePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(factor), &factor);
        vkCmdDispatch(commandBuffer, (m_outputExtent.width + 7) / 8, (m_outputExtent.height + 7) / 8, 1);

        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        vkCmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0,
            0,
            nullptr,
            0,
            nullptr,
            1,
            &barrier);
    }

    void GroundTruthPass::Cleanup()
    {
        vkDestroyPipeline(m_device, m_downsamplePipeline, nullptr);
        vkDestroyPipelineLayout(m_device, m_downsamplePipelineLayout, nullptr);
        vkDestroyDescriptorPool(m_device, m_downsamplePool, nullptr);
        vkDestroyDescriptorSetLayout(m_device, m_downsampleSetLayout, nullptr);
        vkDestroyPipeline(m_device, m_pipeline, nullptr);
        vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);

        for (auto framebuffer : m_framebuffers)
        {
            vkDestroyFramebuffer(m_device, framebuffer, nullptr);
        }
        m_framebuffers.clear();

        vkDestroyRenderPass(m_device, m_renderPass, nullptr);

        m_downsamplePipeline = VK_NULL_HANDLE;
        m_downsamplePipelineLayout = VK_NULL_HANDLE;
        m_downsamplePool = VK_NULL_HANDLE;
        m_downsampleSetLayout = VK_NULL_HANDLE;
        m_pipeline = VK_NULL_HANDLE;
        m_pipelineLayout = VK_NULL_HANDLE;
        m_renderPass = VK_NULL_HANDLE;

        DestroyImage(m_output);
        DestroyImage(m_superColor);
        DestroyImage(m_superNormal); // Zwolnienie nowego zasobu normalnych
        DestroyImage(m_superDepth);
    }
}