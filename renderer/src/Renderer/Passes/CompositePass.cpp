#include "CompositePass.hpp"

#include <stdexcept>
#include "Utils/FileUtils.hpp"

namespace Engine::Renderer
{
    CompositePass::CompositePass(VulkanContext &vulkanContext)
        : m_vulkanContext(vulkanContext) {}

    CompositePass::~CompositePass()
    {
        cleanup();
    }

    void CompositePass::init()
    {
        createBuffers();
        createDescriptors();
        createPipeline();
    }

    void CompositePass::cleanup()
    {
        VkDevice device = m_vulkanContext.getDevice();
        VmaAllocator allocator = m_vulkanContext.getAllocator();

        for (int i = 0; i < BUFFER_COUNT; i++)
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

        if (m_compositePipeline != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(device, m_compositePipeline, nullptr);
            m_compositePipeline = VK_NULL_HANDLE;
        }
        if (m_compositePipelineLayout != VK_NULL_HANDLE)
        {
            vkDestroyPipelineLayout(device, m_compositePipelineLayout, nullptr);
            m_compositePipelineLayout = VK_NULL_HANDLE;
        }
        if (m_compositePool != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorPool(device, m_compositePool, nullptr);
            m_compositePool = VK_NULL_HANDLE;
        }
        if (m_compositeDescriptorLayout != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorSetLayout(device, m_compositeDescriptorLayout, nullptr);
            m_compositeDescriptorLayout = VK_NULL_HANDLE;
        }
    }

    void CompositePass::createBuffers()
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

        for (int i = 0; i < BUFFER_COUNT; i++)
        {
            vmaCreateImage(allocator, &imageInfo, &allocInfo, &m_buffers[i].image, &m_buffers[i].allocation, nullptr);

            VkImageViewCreateInfo viewInfo{};
            viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            viewInfo.image = m_buffers[i].image;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            viewInfo.subresourceRange.levelCount = 1;
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

    void CompositePass::createDescriptors()
    {
        VkDevice device = m_vulkanContext.getDevice();

        VkDescriptorSetLayoutBinding bindings[BUFFER_COUNT];
        for (int i = 0; i < BUFFER_COUNT; i++)
        {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            bindings[i].pImmutableSamplers = nullptr;
        }

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = BUFFER_COUNT;
        layoutInfo.pBindings = bindings;
        vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &m_compositeDescriptorLayout);

        VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, BUFFER_COUNT};
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        poolInfo.maxSets = 1;
        vkCreateDescriptorPool(device, &poolInfo, nullptr, &m_compositePool);

        VkDescriptorSetAllocateInfo setAllocInfo{};
        setAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        setAllocInfo.descriptorPool = m_compositePool;
        setAllocInfo.descriptorSetCount = 1;
        setAllocInfo.pSetLayouts = &m_compositeDescriptorLayout;
        vkAllocateDescriptorSets(device, &setAllocInfo, &m_compositeDescriptorSet);

        VkDescriptorImageInfo imageInfos[BUFFER_COUNT];
        VkWriteDescriptorSet descriptorWrites[BUFFER_COUNT] = {};
        for (int i = 0; i < BUFFER_COUNT; i++)
        {
            imageInfos[i].sampler = m_buffers[i].sampler;
            imageInfos[i].imageView = m_buffers[i].imageView;
            imageInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[i].dstSet = m_compositeDescriptorSet;
            descriptorWrites[i].dstBinding = i;
            descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[i].descriptorCount = 1;
            descriptorWrites[i].pImageInfo = &imageInfos[i];
        }
        vkUpdateDescriptorSets(device, BUFFER_COUNT, descriptorWrites, 0, nullptr);
    }

    void CompositePass::createPipeline()
    {
        VkDevice device = m_vulkanContext.getDevice();

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &m_compositeDescriptorLayout;
        vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &m_compositePipelineLayout);

        auto vertShaderCode = Utils::readFile("shaders/composite.vert.spv");
        auto fragShaderCode = Utils::readFile("shaders/composite.frag.spv");
        VkShaderModule vertModule = createShaderModule(vertShaderCode);
        VkShaderModule fragModule = createShaderModule(fragShaderCode);

        VkPipelineShaderStageCreateInfo shaderStages[] = {
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vertModule, "main", nullptr},
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fragModule, "main", nullptr}
        };

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
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

        vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_compositePipeline);

        vkDestroyShaderModule(device, vertModule, nullptr);
        vkDestroyShaderModule(device, fragModule, nullptr);
    }

    void CompositePass::recordBufferClears(VkCommandBuffer commandBuffer)
    {
        VkClearColorValue clearColors[BUFFER_COUNT] = {
            {{0.0f, 0.0f, 0.0f, 1.0f}}, // Buffer 0: Overwritten by GeometryPass
            {{0.0f, 1.0f, 0.0f, 1.0f}}, // Green (Normals)
            {{0.0f, 0.0f, 1.0f, 1.0f}}, // Blue (Motion Vectors)
            {{1.0f, 1.0f, 0.0f, 1.0f}}  // Yellow (SSAA Ground Truth)
        };

        for (int i = 0; i < BUFFER_COUNT; i++)
        {
            // Transition: UNDEFINED -> TRANSFER_DST
            insertImageMemoryBarrier(
                commandBuffer, m_buffers[i].image,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                0, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT
            );

            VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(commandBuffer, m_buffers[i].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearColors[i], 1, &range);

            // Transition: TRANSFER_DST -> SHADER_READ_ONLY_OPTIMAL
            insertImageMemoryBarrier(
                commandBuffer, m_buffers[i].image,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
            );
        }
    }

    void CompositePass::drawComposite(VkCommandBuffer commandBuffer)
    {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_compositePipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_compositePipelineLayout, 0, 1, &m_compositeDescriptorSet, 0, nullptr);
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    }

    void CompositePass::insertImageMemoryBarrier(
        VkCommandBuffer commandBuffer,
        VkImage image,
        VkImageLayout oldLayout,
        VkImageLayout newLayout,
        VkAccessFlags srcAccess,
        VkAccessFlags dstAccess,
        VkPipelineStageFlags srcStage,
        VkPipelineStageFlags dstStage
    )
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

        vkCmdPipelineBarrier(commandBuffer, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    }

    VkShaderModule CompositePass::createShaderModule(const std::vector<char> &code)
    {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());

        VkShaderModule shaderModule;
        vkCreateShaderModule(m_vulkanContext.getDevice(), &createInfo, nullptr, &shaderModule);
        return shaderModule;
    }
}