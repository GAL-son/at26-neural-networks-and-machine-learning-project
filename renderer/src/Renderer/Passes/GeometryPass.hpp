#pragma once

#include <vulkan/vulkan.hpp>
#include <glm/glm.hpp>
#include <vector>
#include <string>
#include "VulkanContext.hpp"
#include "../Scene/Scene.hpp"
#include "../Utils/FileUtils.hpp"

class GeometryPass {
public:
    explicit GeometryPass(VulkanContext &vulkanContext);
    ~GeometryPass();

    // Prevent copying
    GeometryPass(const GeometryPass &) = delete;
    GeometryPass &operator=(const GeometryPass &) = delete;

    void init();
    void cleanup();

    // Execution passes
    void render(VkCommandBuffer commandBuffer, const Scene &scene, VkImageView targetImageView, VkImage targetImage, VkExtent2D extent);
    void renderNormals(VkCommandBuffer commandBuffer, const Scene &scene, VkImageView targetImageView, VkImage targetImage, VkExtent2D extent);

    VkPipelineLayout getPipelineLayout() const { return m_pipelineLayout; }

private:
    VulkanContext &m_vulkanContext;

    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_textureDescriptorSetLayout = VK_NULL_HANDLE;

    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkPipeline m_normalPipeline = VK_NULL_HANDLE;

    // Internal helpers
    VkShaderModule createShaderModule(const std::vector<char> &code);
    VkPipeline createPipeline(VkDevice device, const std::string& vertPath, const std::string& fragPath, 
                              const VkPipelineVertexInputStateCreateInfo& vertexInputInfo, 
                              const VkPipelineRenderingCreateInfo& renderingInfo);
    
    void transitionImageLayout(VkCommandBuffer commandBuffer, VkImage image, 
                               VkImageLayout oldLayout, VkImageLayout newLayout,
                               VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage,
                               VkAccessFlags srcAccess, VkAccessFlags dstAccess);
};