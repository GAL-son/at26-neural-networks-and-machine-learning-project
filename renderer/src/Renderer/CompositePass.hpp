#pragma once
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <vector>
#include "VulkanContext.hpp"

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
    CompositePass(VulkanContext &vulkanContext);
    ~CompositePass();

    void init();
    void cleanup();

    // Call this BEFORE vkCmdBeginRendering
    void recordBufferClears(VkCommandBuffer commandBuffer);

    // Call this INSIDE vkCmdBeginRendering
    void drawComposite(VkCommandBuffer commandBuffer);

private:
    VkShaderModule createShaderModule(const std::vector<char> &code);

    VulkanContext &m_vulkanContext;
    OffscreenBuffer m_buffers[4];
    VkDescriptorSetLayout m_compositeDescriptorLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_compositePool = VK_NULL_HANDLE;
    VkDescriptorSet m_compositeDescriptorSet = VK_NULL_HANDLE;
    VkPipelineLayout m_compositePipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_compositePipeline = VK_NULL_HANDLE;
};