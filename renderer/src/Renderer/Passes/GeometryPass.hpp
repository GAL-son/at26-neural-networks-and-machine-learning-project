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
    // Struktura UBO dopasowana do layoutu w shaderze (wyrównanie std140)
    struct MaterialUniformData
    {
        glm::vec3 ambient;
        float shininess;
        glm::vec3 diffuse;
        float opacity;
        glm::vec3 specular;
        int32_t hasDiffuseTexture;
        int32_t pad1; // Wyrównanie do 16 bajtów
        int32_t pad2;
    };

    class GeometryPass : public RenderPass
    {
    public:
        GeometryPass(VulkanContext &context, VkExtent2D extent);
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

        // Zasoby MRT (4 obrazy kolorów)
        std::vector<VkImageView> m_targetImageViews;
        std::vector<VmaAllocation> m_targetAllocations;

        // Bufor głębi
        VkImage m_depthImage{VK_NULL_HANDLE};
        VkImageView m_depthImageView{VK_NULL_HANDLE};
        VmaAllocation m_depthAllocation{VK_NULL_HANDLE};

        // Pipeline i Layout
        VkPipeline m_pipeline{VK_NULL_HANDLE};
        VkPipelineLayout m_pipelineLayout{VK_NULL_HANDLE};
        VkDescriptorSetLayout m_descriptorSetLayout{VK_NULL_HANDLE};
        VkDescriptorPool m_descriptorPool{VK_NULL_HANDLE};

        // Zasoby per-materiał (bufory UBO oraz alokacje descriptor setów dla materiałów ze sceny)
        std::vector<std::unique_ptr<Buffer>> m_materialBuffers;
        std::vector<VkDescriptorSet> m_materialDescriptorSets;

        void CreateAttachments();
        void CreateRenderPassCustom();
        void CreateFramebufferCustom();
        void CreatePipelineCustom();
        void CreateMaterialResources(Renderer::Scene &scene); // Nowa metoda inicjalizująca materiały
    };
}