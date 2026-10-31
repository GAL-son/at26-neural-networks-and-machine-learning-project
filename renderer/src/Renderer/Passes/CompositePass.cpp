#include "CompositePass.hpp"
#include <stdexcept>
#include <array>
#include "Utils/FileUtils.hpp" // lub Twoja metoda wczytywania plików spv

namespace Engine::Renderer
{
    CompositePass::CompositePass(
        VulkanContext& context, 
        VkExtent2D extent, 
        const std::vector<VkImageView>& inputImageViews, 
        VkSampler sampler)
        : RenderPass(context), m_extent(extent), m_inputImageViews(inputImageViews), m_sampler(sampler)
    {
    }

    void CompositePass::Create()
    {
        CreateDescriptorSetLayout();
        CreateDescriptorPool();
        CreateDescriptorSet();
        CreatePipeline();
    }

    void CompositePass::Cleanup()
    {
        if (m_pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(m_device, m_pipeline, nullptr);
            m_pipeline = VK_NULL_HANDLE;
        }
        if (m_pipelineLayout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
            m_pipelineLayout = VK_NULL_HANDLE;
        }
        if (m_descriptorPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
            m_descriptorPool = VK_NULL_HANDLE;
        }
        if (m_descriptorSetLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr);
            m_descriptorSetLayout = VK_NULL_HANDLE;
        }
    }

    void CompositePass::CreateDescriptorSetLayout()
    {
        std::array<VkDescriptorSetLayoutBinding, 4> bindings{};
        for (int i = 0; i < 4; i++)
        {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            bindings[i].pImmutableSamplers = nullptr;
        }

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_descriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create composite descriptor set layout!");
        }
    }

    void CompositePass::CreateDescriptorPool()
    {
        VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4};

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        poolInfo.maxSets = 1;

        if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create composite descriptor pool!");
        }
    }

    void CompositePass::CreateDescriptorSet()
    {
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = m_descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &m_descriptorSetLayout;

        if (vkAllocateDescriptorSets(m_device, &allocInfo, &m_descriptorSet) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate composite descriptor set!");
        }

        std::array<VkDescriptorImageInfo, 4> imageInfos{};
        std::array<VkWriteDescriptorSet, 4> descriptorWrites{};

        for (size_t i = 0; i < 4; i++)
        {
            imageInfos[i].sampler = m_sampler;
            imageInfos[i].imageView = m_inputImageViews[i];
            imageInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[i].dstSet = m_descriptorSet;
            descriptorWrites[i].dstBinding = static_cast<uint32_t>(i);
            descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[i].descriptorCount = 1;
            descriptorWrites[i].pImageInfo = &imageInfos[i];
        }

        vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }

    void CompositePass::CreatePipeline()
    {
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &m_descriptorSetLayout;

        if (vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, &m_pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create composite pipeline layout!");
        }

        // Wczytywanie skompilowanych shaderów kompozycji
        auto vertShaderCode = Utils::readFile("shaders/composite.vert.spv");
        auto fragShaderCode = Utils::readFile("shaders/composite.frag.spv");
        
        // Zakładamy, że masz metodę pomocniczą do tworzenia modułów shaderów
        // (możesz ją przenieść lub użyć swojej fabryki)
        auto createShaderModule = [this](const std::vector<char>& code) {
            VkShaderModuleCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            createInfo.codeSize = code.size();
            createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
            VkShaderModule module;
            vkCreateShaderModule(m_device, &createInfo, nullptr, &module);
            return module;
        };

        VkShaderModule vertModule = createShaderModule(vertShaderCode);
        VkShaderModule fragModule = createShaderModule(fragShaderCode);

        VkPipelineShaderStageCreateInfo shaderStages[] = {
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vertModule, "main", nullptr},
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fragModule, "main", nullptr}
        };

        // Brak vertex input, bo używamy pełnoekranowego trójkąta generowanego w shaderze!
        VkPipelineVertexInputStateCreateInfo vertexInputInfo{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        
        VkPipelineInputAssemblyStateCreateInfo inputAssembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo viewportState{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rasterizer{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;

        VkPipelineMultisampleStateCreateInfo multisampling{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo colorBlending{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;

        std::vector<VkDynamicState> dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamicState{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        // Używamy dynamic rendering dla wyjścia na swapchain (lub standardowego RenderPassa, jeśli Twój silnik go wymaga)
        VkFormat swapchainFormat = m_context.getSwapchainImageFormat();
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
        pipelineInfo.layout = m_pipelineLayout;

        if (vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pipeline) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create composite pipeline!");
        }

        vkDestroyShaderModule(m_device, fragModule, nullptr);
    }

    void CompositePass::Begin(VkCommandBuffer commandBuffer, VkExtent2D extent)
    {
        // Tutaj rozpoczynamy renderowanie na docelowy obraz (np. swapchain frame)
        // Jeśli używasz dynamic rendering, konfigurujesz strukturę vkCmdBeginRendering.
    }

    void CompositePass::Render(VkCommandBuffer commandBuffer, Renderer::Scene& scene, Camera& camera)
    {
        // 1. Podpinamy pipeline kompozycji
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);

        // 2. Podpinamy zestaw deskryptorów zawierający 4 bufory z GeometryPass
        vkCmdBindDescriptorSets(
            commandBuffer, 
            VK_PIPELINE_BIND_POINT_GRAPHICS, 
            m_pipelineLayout, 
            0, 
            1, 
            &m_descriptorSet, 
            0, 
            nullptr
        );

        // 3. Rysujemy pełnoekranowy trójkąt (3 wierzchołki generowane w shaderze)
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    }

    void CompositePass::End(VkCommandBuffer commandBuffer)
    {
        // Zakończenie renderowania
    }
}