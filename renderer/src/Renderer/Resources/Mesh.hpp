#pragma once

#include <memory>
#include <vulkan/vulkan.hpp>
#include "Scene/Scene.hpp"
#include "Buffer.hpp"
#include "VulkanContext.hpp"

namespace Engine::Renderer
{
    class Mesh
    {
    public:
        // Tworzy zasoby GPU na podstawie danych z CPU
        Mesh(VulkanContext& context, const Scene::Mesh& cpuMesh);
        ~Mesh() = default;

        // Samobindowanie się do bufora poleceń (zgodnie z Twoim założeniem!)
        void Bind(VkCommandBuffer commandBuffer) const;

        uint32_t GetIndexCount() const { return m_indexCount; }
        const std::vector<Scene::Submesh>& GetSubmeshes() const { return m_submeshes; }

    private:
        std::unique_ptr<Buffer> m_vertexBuffer;
        std::unique_ptr<Buffer> m_indexBuffer;
        
        uint32_t m_indexCount{0};
        std::vector<Scene::Submesh> m_submeshes;
    };
}