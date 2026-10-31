#include "Scene.hpp"

#include "Renderer/Resources/Mesh.hpp"
#include "Renderer/Resources/Texture.hpp"

namespace Engine::Renderer
{
    Scene::Scene(VulkanContext &context, VkCommandPool commandPool, const Engine::Scene::Scene &cpuScene)
        : m_allocator(context.getAllocator())
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
}