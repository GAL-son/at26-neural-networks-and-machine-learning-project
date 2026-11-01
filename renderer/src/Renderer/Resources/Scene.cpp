#include "Scene.hpp"

#include "Renderer/Resources/Mesh.hpp"
#include "Renderer/Resources/Texture.hpp"

namespace Engine::Renderer
{
    Scene::Scene(
        VulkanContext &context,
        VkCommandPool commandPool,
        const Engine::Scene::Scene &cpuScene,
        VkDescriptorSetLayout materialSetLayout)
        : m_allocator(context.getAllocator()),
          m_device(context.getDevice())
    {
        // 1. Konwersja i upload wszystkich tekstur na GPU
        m_gpuTextures.reserve(cpuScene.textures.size());
        for (const auto &cpuTex : cpuScene.textures)
        {
            m_gpuTextures.push_back(std::make_unique<Texture>(context, commandPool, cpuTex));
        }

        // 2. Skopiowanie czystych danych materiałowych z CPU do GPU Scene
        // Brak alokacji deskryptorów Vulcana! Materiały to teraz zwykłe dane parametrów.
        m_gpuMaterials = cpuScene.materials;

        // 3. Konwersja i upload wszystkich meshów na GPU
        m_gpuMeshes.reserve(cpuScene.meshes.size());
        for (const auto &cpuMesh : cpuScene.meshes)
        {
            m_gpuMeshes.push_back(std::make_unique<Mesh>(m_allocator, cpuMesh));
        }
    }

    Scene::~Scene()
    {
        m_materialBuffers.clear();
        if (m_materialPool != VK_NULL_HANDLE)
            vkDestroyDescriptorPool(m_device, m_materialPool, nullptr);
    }

    VkDescriptorSet Scene::GetMaterialDescriptorSet(int materialIndex) const
    {
        if (materialIndex < 0 || static_cast<size_t>(materialIndex) >= m_materialBuffers.size())
            materialIndex = 0;

        return m_materialSets[materialIndex];
    }

    void Scene::Draw(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout) const
    {
        for (const auto &mesh : m_gpuMeshes)
        {
            mesh->Bind(commandBuffer);
            for (const Engine::Scene::Submesh &submesh : mesh->GetSubmeshes())
            {
                VkDescriptorSet materialSet = GetMaterialDescriptorSet(submesh.materialIndex);
                vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &materialSet, 0, nullptr);
                vkCmdDrawIndexed(commandBuffer, submesh.indexCount, 1, submesh.indexOffset, 0, 0);
            }
        }
    }
}