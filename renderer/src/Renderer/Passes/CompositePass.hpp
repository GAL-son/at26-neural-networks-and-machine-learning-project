#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <vector>
#include "VulkanContext.hpp"

namespace Engine::Renderer
{
    struct OffscreenBuffer
    {
        VkImage image = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        VkImageView imageView = VK_NULL_HANDLE;
        VkSampler sampler = VK_NULL_HANDLE;
    };

    class CompositePass
    {
    public:
        static constexpr int BUFFER_COUNT = 4;

        explicit CompositePass(VulkanContext &vulkanContext);
        ~CompositePass();

        void init();
        void cleanup();

        // Call this BEFORE vkCmdBeginRendering
        void recordBufferClears(VkCommandBuffer commandBuffer);

        // Call this INSIDE vkCmdBeginRendering
        void drawComposite(VkCommandBuffer commandBuffer);

        const OffscreenBuffer& getBuffer(int index) const { return m_buffers[index]; }

    private:
        void createBuffers();
        void createDescriptors();
        void createPipeline();
        
        VkShaderModule createShaderModule(const std::vector<char> &code);
        
        void insertImageMemoryBarrier(
            VkCommandBuffer commandBuffer,
            VkImage image,
            VkImageLayout oldLayout,
            VkImageLayout newLayout,
            VkAccessFlags srcAccess,
            VkAccessFlags dstAccess,
            VkPipelineStageFlags srcStage,
            VkPipelineStageFlags dstStage
        );

        VulkanContext &m_vulkanContext;
        OffscreenBuffer m_buffers[BUFFER_COUNT];
        
        VkDescriptorSetLayout m_compositeDescriptorLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_compositePool = VK_NULL_HANDLE;
        VkDescriptorSet m_compositeDescriptorSet = VK_NULL_HANDLE;
        VkPipelineLayout m_compositePipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_compositePipeline = VK_NULL_HANDLE;
    };
}