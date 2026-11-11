
#pragma once

#include <vk_mem_alloc.h>
#include "RenderPass.hpp"

namespace Engine::Renderer
{
    // SSAA Ground Truth pass
    // Renders the same way as Geometry pass but with super sampling and without extra buffers (albedo only)
    class GroundTruthPass : public RenderPass
    {
    public:
        GroundTruthPass(VulkanContext &context, VkExtent2D outputExtent, uint32_t ssaaFactor, VkDescriptorSetLayout materialSetLayout);
        ~GroundTruthPass() override = default;

        void Create() override;
        void Cleanup() override;

        void Begin(VkCommandBuffer commandBuffer, VkExtent2D /*extent*/) override;
        void Render(VkCommandBuffer commandBuffer, Renderer::Scene &scene, Camera &camera) override;
        void End(VkCommandBuffer commandBuffer) override;

        VkImage GetOutputImage() const { return m_output.image; };
        VkImageView GetOutputView() const { return m_output.view; };
        VkExtent2D getOutputExtent() const { return m_outputExtent; };

    private:
        struct AllocatedImage
        {
            VkImage image{VK_NULL_HANDLE};
            VkImageView view{VK_NULL_HANDLE};
            VmaAllocation allocation{VK_NULL_HANDLE};
        };

        VkExtent2D m_outputExtent;
        VkExtent2D m_superExtent;
        uint32_t m_ssaaFactor;

        AllocatedImage m_superColor;
        AllocatedImage m_superNormal;
        AllocatedImage m_superDepth;
        AllocatedImage m_output;

        VkPipelineLayout m_pipelineLayout{VK_NULL_HANDLE};
        VkPipeline m_pipeline{VK_NULL_HANDLE};
        VkDescriptorSetLayout m_materialSetLayout{VK_NULL_HANDLE};

        VkDescriptorSetLayout m_downsampleSetLayout{VK_NULL_HANDLE};
        VkDescriptorPool m_downsamplePool{VK_NULL_HANDLE};
        VkDescriptorSet m_downsampleSet{VK_NULL_HANDLE};
        VkPipelineLayout m_downsamplePipelineLayout{VK_NULL_HANDLE};
        VkPipeline m_downsamplePipeline{VK_NULL_HANDLE};

        void ValidateLimits() const;
        AllocatedImage CreateImage(VkFormat format, VkExtent2D extent, VkImageUsageFlags usage, VkImageAspectFlags aspect);
        void DestroyImage(AllocatedImage &image);

        void CreateRenderPassCustom();
        void CreateFramebufferCustom();
        void CreateGraphicsPipeline();
        void CreateDownsamplePipeline();
    };
}