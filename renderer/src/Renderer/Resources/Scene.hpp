#pragma once

#include <vector>
#include <memory>
#include <vk_mem_alloc.h>
#include "Scene/Scene.hpp"
#include "Renderer/Resources/Mesh.hpp"
#include "Renderer/Resources/Texture.hpp"

namespace Engine::Renderer
{
    class Scene
    {
    public:
        // Konstruktor przyjmuje kontekst, pulę komend oraz surową scenę z CPU
        Scene(VulkanContext& context, VkCommandPool commandPool, const Engine::Scene::Scene& cpuScene);
        ~Scene() = default;

        // Dostęp do zasobów GPU oraz danych materiałowych dla pasów renderowania
        const std::vector<std::unique_ptr<Mesh>>& GetMeshes() const { return m_gpuMeshes; }
        const std::vector<std::unique_ptr<Texture>>& GetTextures() const { return m_gpuTextures; }
        const std::vector<Engine::Scene::Material>& GetMaterials() const { return m_gpuMaterials; }

    private:
        VmaAllocator m_allocator;

        // Zasoby GPU i czyste dane materiałowe
        std::vector<std::unique_ptr<Mesh>> m_gpuMeshes;
        std::vector<Engine::Scene::Material> m_gpuMaterials; // Czyste dane mtl (kolory, indeksy tekstur)
        std::vector<std::unique_ptr<Texture>> m_gpuTextures;
    };
}