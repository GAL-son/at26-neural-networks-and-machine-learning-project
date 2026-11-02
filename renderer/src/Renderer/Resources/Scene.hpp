#pragma once

#include <vector>
#include <memory>
#include <vk_mem_alloc.h>
#include "Scene/Scene.hpp"
#include "Renderer/Resources/Mesh.hpp"
#include "Renderer/Resources/Texture.hpp"

namespace Engine::Renderer
{
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

    class Scene
    {
    public:
        // Konstruktor przyjmuje kontekst, pulę komend oraz surową scenę z CPU
        Scene(VulkanContext& context, VkCommandPool commandPool, const Engine::Scene::Scene& cpuScene, VkDescriptorSetLayout materialSetLayout);
        ~Scene();

        VkDescriptorSet GetMaterialDescriptorSet(int materialIndex) const;

        void Draw(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLeyout) const;

        // Dostęp do zasobów GPU oraz danych materiałowych dla pasów renderowania
        const std::vector<std::unique_ptr<Mesh>>& GetMeshes() const { return m_gpuMeshes; }
        const std::vector<std::unique_ptr<Texture>>& GetTextures() const { return m_gpuTextures; }
        const std::vector<Engine::Scene::Material>& GetMaterials() const { return m_gpuMaterials; }

    private:
        VkDevice m_device;
        VkDescriptorPool m_materialPool{VK_NULL_HANDLE};
        std::vector<std::unique_ptr<Buffer>> m_materialBuffers;
        std::vector<VkDescriptorSet> m_materialSets;

        VmaAllocator m_allocator;

        // Zasoby GPU i czyste dane materiałowe
        std::vector<std::unique_ptr<Mesh>> m_gpuMeshes;
        std::vector<Engine::Scene::Material> m_gpuMaterials; // Czyste dane mtl (kolory, indeksy tekstur)
        std::vector<std::unique_ptr<Texture>> m_gpuTextures;

        std::unique_ptr<Texture> m_fallbackTexture;

        void CreateMaterialResources(VkDescriptorSetLayout materialSetLayout);
    };
}