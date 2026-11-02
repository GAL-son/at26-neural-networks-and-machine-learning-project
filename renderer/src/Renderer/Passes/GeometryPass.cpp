#include "GeometryPass.hpp"
#include <stdexcept>
#include <array>

#include "Utils/FileUtils.hpp"
#include "Renderer/Resources/Texture.hpp"

namespace Engine::Renderer
{
    GeometryPass::GeometryPass(VulkanContext &context, VkExtent2D extent, VkDescriptorSetLayout materialSetLayout)
        : RenderPass(context), m_extent(extent), m_materialSetLayout(materialSetLayout)
    {
    }

    void GeometryPass::Create()
    {
        CreateAttachments();
        CreateRenderPassCustom();
        CreateFramebufferCustom();
        CreatePipelineCustom();
    }

    void GeometryPass::Cleanup()
    {
        if (m_pipeline != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(m_device, m_pipeline, nullptr);
            m_pipeline = VK_NULL_HANDLE;
        }

        if (m_pipelineLayout != VK_NULL_HANDLE)
        {
            vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
            m_pipelineLayout = nullptr;
        }

        if (!m_framebuffers.empty())
        {
            for (auto fb : m_framebuffers)
            {
                vkDestroyFramebuffer(m_device, fb, nullptr);
            }
            m_framebuffers.clear();
        }

        if (m_renderPass != VK_NULL_HANDLE)
        {
            vkDestroyRenderPass(m_device, m_renderPass, nullptr);
            m_renderPass = VK_NULL_HANDLE;
        }

        // Czyszczenie tekstur MRT przez VMA
        for (size_t i = 0; i < m_targetImages.size(); ++i)
        {
            if (m_targetImageViews[i])
            {
                vkDestroyImageView(m_device, m_targetImageViews[i], nullptr);
                m_targetImageViews[i] = VK_NULL_HANDLE;
            }
            if (m_targetImages[i] && m_targetAllocations[i])
            {
                vmaDestroyImage(m_context.getAllocator(), m_targetImages[i], m_targetAllocations[i]);
                m_targetImages[i] = VK_NULL_HANDLE;
                m_targetAllocations[i] = VK_NULL_HANDLE;
            }
        }
        m_targetImages.clear();
        m_targetImageViews.clear();
        m_targetAllocations.clear();

        // 2. Czyszczenie głębi (upewnij się, że te linie faktycznie się wykonują)
        if (m_depthImageView != VK_NULL_HANDLE)
        {
            vkDestroyImageView(m_device, m_depthImageView, nullptr);
            m_depthImageView = VK_NULL_HANDLE;
        }
        if (m_depthImage != VK_NULL_HANDLE && m_depthAllocation != VK_NULL_HANDLE)
        {
            vmaDestroyImage(m_context.getAllocator(), m_depthImage, m_depthAllocation);
            m_depthImage = VK_NULL_HANDLE;
            m_depthAllocation = VK_NULL_HANDLE;
        }
    }

    void GeometryPass::CreateAttachments()
    {
        // Teraz mamy tylko 3 formaty kolorów (Color, Normal, Motion Vectors)
        const std::array<VkFormat, 3> formats = {
            VK_FORMAT_R8G8B8A8_UNORM,      // 0: Color
            VK_FORMAT_R16G16B16A16_SFLOAT, // 1: Normal
            VK_FORMAT_R16G16_SFLOAT        // 2: Motion Vectors
        };

        m_targetImages.resize(3);
        m_targetImageViews.resize(3);
        m_targetAllocations.resize(3);

        for (size_t i = 0; i < 3; ++i)
        {
            VkImageCreateInfo imageInfo{};
            imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            imageInfo.imageType = VK_IMAGE_TYPE_2D;
            imageInfo.extent.width = m_extent.width;
            imageInfo.extent.height = m_extent.height;
            imageInfo.extent.depth = 1;
            imageInfo.mipLevels = 1;
            imageInfo.arrayLayers = 1;
            imageInfo.format = formats[i];
            imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            VmaAllocationCreateInfo allocInfo{};
            allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

            vmaCreateImage(m_context.getAllocator(), &imageInfo, &allocInfo, &m_targetImages[i], &m_targetAllocations[i], nullptr);

            VkImageViewCreateInfo viewInfo{};
            viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            viewInfo.image = m_targetImages[i];
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = formats[i];
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            viewInfo.subresourceRange.baseMipLevel = 0;
            viewInfo.subresourceRange.levelCount = 1;
            viewInfo.subresourceRange.baseArrayLayer = 0;
            viewInfo.subresourceRange.layerCount = 1;

            vkCreateImageView(m_device, &viewInfo, nullptr, &m_targetImageViews[i]);
        }

        // Tworzenie bufora głębi (pozostaje bez zmian, z flagą SAMPLED_BIT)
        VkImageCreateInfo depthImageInfo{};
        depthImageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        depthImageInfo.imageType = VK_IMAGE_TYPE_2D;
        depthImageInfo.extent.width = m_extent.width;
        depthImageInfo.extent.height = m_extent.height;
        depthImageInfo.extent.depth = 1;
        depthImageInfo.mipLevels = 1;
        depthImageInfo.arrayLayers = 1;
        depthImageInfo.format = VK_FORMAT_D32_SFLOAT;
        depthImageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        depthImageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthImageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT; // <-- DODANE
        depthImageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        depthImageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo depthAllocInfo{};
        depthAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

        vmaCreateImage(m_context.getAllocator(), &depthImageInfo, &depthAllocInfo, &m_depthImage, &m_depthAllocation, nullptr);

        VkImageViewCreateInfo depthViewInfo{};
        depthViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        depthViewInfo.image = m_depthImage;
        depthViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        depthViewInfo.format = VK_FORMAT_D32_SFLOAT;
        depthViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        depthViewInfo.subresourceRange.baseMipLevel = 0;
        depthViewInfo.subresourceRange.levelCount = 1;
        depthViewInfo.subresourceRange.baseArrayLayer = 0;
        depthViewInfo.subresourceRange.layerCount = 1;

        vkCreateImageView(m_device, &depthViewInfo, nullptr, &m_depthImageView);
    }

    void GeometryPass::CreateRenderPassCustom()
    {
        // 3 bufory kolorów + 1 bufor głębi = razem 4 attachmenty (indekse 0, 1, 2, 3)
        std::array<VkAttachmentDescription, 4> attachments{};
        const std::array<VkFormat, 3> formats = {
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_FORMAT_R16G16_SFLOAT};

        // Konfiguracja 3 attachmentów kolorów (indeksy 0, 1, 2)
        for (int i = 0; i < 3; ++i)
        {
            attachments[i].format = formats[i];
            attachments[i].samples = VK_SAMPLE_COUNT_1_BIT;
            attachments[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            attachments[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            attachments[i].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            attachments[i].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachments[i].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            attachments[i].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }

        // 4. Depth buffer
        attachments[3].format = VK_FORMAT_D32_SFLOAT;
        attachments[3].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[3].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[3].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[3].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[3].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[3].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[3].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

        std::array<VkAttachmentReference, 3> colorReferences = {
            VkAttachmentReference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
            VkAttachmentReference{1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
            VkAttachmentReference{2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}};

        // Głębokość wskazuje teraz na indeks 3
        VkAttachmentReference depthReference{3, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = static_cast<uint32_t>(colorReferences.size());
        subpass.pColorAttachments = colorReferences.data();
        subpass.pDepthStencilAttachment = &depthReference;

        VkSubpassDependency dependency{};
        dependency.srcSubpass = 0;
        dependency.dstSubpass = VK_SUBPASS_EXTERNAL;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dependency.dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        dependency.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dependency.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(m_device, &renderPassInfo, nullptr, &m_renderPass) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create MRT Geometry RenderPass!");
        }
    }

    void GeometryPass::CreateFramebufferCustom()
    {
        // 3 widoki kolorów + 1 widok głębi (całość ma rozmiar 4)
        std::array<VkImageView, 4> views = {
            m_targetImageViews[0],
            m_targetImageViews[1],
            m_targetImageViews[2],
            m_depthImageView};

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = m_renderPass;
        framebufferInfo.attachmentCount = static_cast<uint32_t>(views.size());
        framebufferInfo.pAttachments = views.data();
        framebufferInfo.width = m_extent.width;
        framebufferInfo.height = m_extent.height;
        framebufferInfo.layers = 1;

        m_framebuffers.resize(1);
        if (vkCreateFramebuffer(m_device, &framebufferInfo, nullptr, &m_framebuffers[0]) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create Geometry Framebuffer!");
        }
    }

    void GeometryPass::CreatePipelineCustom()
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

        auto vertShaderCode = Utils::readFile("shaders/geometry.vert.spv");
        auto fragShaderCode = Utils::readFile("shaders/geometry.frag.spv");

        auto createShaderModule = [this](const std::vector<char> &code)
        {
            VkShaderModuleCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            createInfo.codeSize = code.size();
            createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());
            VkShaderModule module;
            vkCreateShaderModule(m_device, &createInfo, nullptr, &module);
            return module;
        };

        VkShaderModule vertModule = createShaderModule(vertShaderCode);
        VkShaderModule fragModule = createShaderModule(fragShaderCode);

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

        std::array<VkPipelineColorBlendAttachmentState, 3> colorBlendAttachments{};
        for (auto &attachment : colorBlendAttachments)
        {
            attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            attachment.blendEnable = VK_FALSE;
        }

        VkPipelineColorBlendStateCreateInfo colorBlending{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        colorBlending.attachmentCount = static_cast<uint32_t>(colorBlendAttachments.size()); // Wyniesie 3
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

    void GeometryPass::Begin(VkCommandBuffer commandBuffer, VkExtent2D extent)
    {
        std::array<VkClearValue, 4> clearValues{};
        clearValues[0].color = {{0.1f, 0.1f, 0.1f, 1.0f}}; // Color
        clearValues[1].color = {{0.0f, 0.0f, 0.0f, 1.0f}}; // Normal
        clearValues[2].color = {{0.0f, 0.0f, 0.0f, 1.0f}}; // Motion Vectors
        clearValues[3].depthStencil = {1.0f, 0};           // Depth (indeks 3)

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = m_renderPass;
        renderPassInfo.framebuffer = m_framebuffers[0];
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = m_extent;
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport viewport{0.0f, 0.0f, static_cast<float>(m_extent.width), static_cast<float>(m_extent.height), 0.0f, 1.0f};
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{{0, 0}, m_extent};
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
    }

    void GeometryPass::Render(VkCommandBuffer commandBuffer, Renderer::Scene &scene, Camera &camera)
    {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
        glm::mat4 vp = camera.getProjectionMatrix() * camera.getViewMatrix();
        glm::mat4 prevVp = camera.getPreviousVpMatrix();

        // Pakujemy obie macierze do struktury Push Constants
        struct PushData
        {
            glm::mat4 vp;
            glm::mat4 prevVp;
        } pushData{vp, prevVp};

        vkCmdPushConstants(
            commandBuffer,
            m_pipelineLayout,
            VK_SHADER_STAGE_VERTEX_BIT,
            0,
            sizeof(PushData),
            &pushData);

        scene.Draw(commandBuffer, m_pipelineLayout);
    }

    void GeometryPass::End(VkCommandBuffer commandBuffer)
    {
        vkCmdEndRenderPass(commandBuffer);
    }
}