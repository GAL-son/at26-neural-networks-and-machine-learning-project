#pragma once

#define VULKAN_HPP_NO_CONSTRUCTORS
#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.h>
#include <vector>
#include "VulkanContext.hpp"
#include "../Scene/Scene.hpp"

class GeometryPass {
public:
    GeometryPass(VulkanContext& vulkanContext);
    ~GeometryPass();

    void init();
    void cleanup();

    // Renders the scene into the specified offscreen target image view using Dynamic Rendering
    void render(VkCommandBuffer commandBuffer, const Scene& scene, VkImageView targetImageView, VkImage targetImage, VkExtent2D extent);
    void renderNormals(VkCommandBuffer commandBuffer, const Scene &scene, VkImageView targetImageView, VkImage targetImage, VkExtent2D extent);

private:
    VulkanContext& m_vulkanContext;

    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkPipelineLayout m_normalPipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_normalPipeline = VK_NULL_HANDLE;

    VkShaderModule createShaderModule(const std::vector<char>& code);

};