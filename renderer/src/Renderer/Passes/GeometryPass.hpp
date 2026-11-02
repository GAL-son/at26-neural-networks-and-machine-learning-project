#pragma once

#include <vector>
#include <memory>
#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.h>

#include "RenderPass.hpp"
#include "Renderer/Resources/Scene.hpp"
#include "Renderer/Resources/Buffer.hpp"
#include "Core/Camera.hpp"

namespace Engine::Renderer
{
    class GeometryPass : public RenderPass
    {
    public:
        GeometryPass(VulkanContext &context, VkExtent2D extent, VkDescriptorSetLayout materialSetLayout);
        ~GeometryPass() override = default;

        void Create() override;
        void Cleanup() override;

        void Begin(VkCommandBuffer commandBuffer, VkExtent2D extent) override;
        void Render(VkCommandBuffer commandBuffer, Renderer::Scene &scene, Camera &camera) override;
        void End(VkCommandBuffer commandBuffer) override;

        // Gettery do tekstur wyjściowych (dla CompositePass)
        const std::vector<VkImage> &GetTargetImages() const { return m_targetImages; }
        const std::vector<VkImageView> &GetTargetImageViews() const { return m_targetImageViews; }
        VkImageView GetDepthImageView() const { return m_depthImageView; }

    private:
        VkExtent2D m_extent;

        // Render Targets
        std::vector<VkImageView> m_targetImageViews;
        std::vector<VmaAllocation> m_targetAllocations;

        // Bufor głębi
        VkImage m_depthImage{VK_NULL_HANDLE};
        VkImageView m_depthImageView{VK_NULL_HANDLE};
        VmaAllocation m_depthAllocation{VK_NULL_HANDLE};

        // Pipeline i Layout
        VkPipelineLayout m_pipelineLayout{VK_NULL_HANDLE};
        VkPipeline m_pipeline{VK_NULL_HANDLE};
        VkDescriptorSetLayout m_materialSetLayout{VK_NULL_HANDLE};

        void CreateAttachments();
        void CreateRenderPassCustom();
        void CreateFramebufferCustom();
        void CreatePipelineCustom();
    };
}