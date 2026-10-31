#pragma once

#include <vector>
#include <vulkan/vulkan.hpp>
#include "RenderPass.hpp"

namespace Engine::Renderer
{
    class CompositePass : public RenderPass
    {
    public:
        // Przyjmuje kontekst, rozmiar oraz widoki 4 buforów z GeometryPass oraz wspólny sampler
        CompositePass(
            VulkanContext& context, 
            VkExtent2D extent, 
            const std::vector<VkImageView>& inputImageViews, 
            VkSampler sampler
        );
        ~CompositePass() override = default;

        void Create() override;
        void Cleanup() override;

        void Begin(VkCommandBuffer commandBuffer, VkExtent2D extent) override;
        // Ponieważ kompozycja nie potrzebuje sceny ani kamery, parametry te pozostają puste
        void Render(VkCommandBuffer commandBuffer, Renderer::Scene& scene, Camera& camera) override;
        void End(VkCommandBuffer commandBuffer) override;

    private:
        VkExtent2D m_extent;
        std::vector<VkImageView> m_inputImageViews;
        VkSampler m_sampler;

        VkPipeline m_pipeline{VK_NULL_HANDLE};
        VkPipelineLayout m_pipelineLayout{VK_NULL_HANDLE};
        VkDescriptorSetLayout m_descriptorSetLayout{VK_NULL_HANDLE};
        VkDescriptorPool m_descriptorPool{VK_NULL_HANDLE};
        VkDescriptorSet m_descriptorSet{VK_NULL_HANDLE};

        void CreateDescriptorSetLayout();
        void CreateDescriptorPool();
        void CreateDescriptorSet();
        void CreatePipeline();
    };
}