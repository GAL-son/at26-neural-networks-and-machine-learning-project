
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

        void Begin(VkCommandBuffer commandBuffer, VkExtent2D extent) override;
        void Render(VkCommandBuffer commandBuffer, Renderer::Scene &scene, Camera &camera) override;
        void End(VkCommandBuffer commandBuffer) override;

        VkImage GetOutputImage() const { return m_output.image };
        VkImageView GetOutputView() const { return m_output.view };
        VkExtent2D getOutputExtent() const { return m_outputExtent };

    private:
        struct AllocatedImage
        {
            VkImage image{VK_NULL_HANDLE};
            VkImageView view{VK_NULL_HANDLE};
            VmaAllocation allocation{VK_NULL_HANDLE};
        };

        VkExtent2D m_outputExtent;
        VkExtent2D m_superExtend;
        uint32_t m_ssaaFactor;
        VkDescriptorSetLayout m_materialSetLayout;

        AllocatedImage m_superColor;
        AllocatedImage m_superDepth;
        AllocatedImage m_output;

        VkDescriptorSetLayout m_downsampleSetLayout{VK_NULL_HANDLE};
        VkDescriptorPool m_downsamlePool{VK_NULL_HANDLE};
        VkDescriptorSet m_downsamleSet{VK_NULL_HANDLE};
        VkPipelineLayout m_downsamplePipelineLayout{VK_NULL_HANDLE};
        VkPipeline m_downsamplePipeline{VK_NULL_HANDLE};

        void ValidateLimits() const;
        AllocatedImage CreateImage(VkFormat format, VkExtent2D extent, VkImageUsageFlags usage, VkImageAspectFlags aspect);
        void DestroyImage(AllocatedImage &image);
        VkShaderModule LoadShader(const char *path);

        void CreateAttachments();
        void CreateRenderPassCustom();
        void CreateFramebufferCustom();
        void CreatePipelineCustom();
    }
}