#pragma once

#include <vector>
#include <memory>
#include <vk_mem_alloc.h>
#include "Scene/Scene.hpp"
#include "Renderer/Resources/Mesh.hpp"

namespace Engine::Renderer
{
    struct GpuRenderable
    {
        std::unique_ptr<Mesh> mesh;
        std::vector<Engine::Scene::Submesh> submeshes;
        glm::mat4 transform;
        // Material or texture references can go here
    };

    class VulkanScene
    {
    public:
        VulkanScene(VmaAllocator allocator);

        void uploadScene(const Engine::Scene::Scene &cpuScene);
        const std::vector<GpuRenderable>& getRenderables() const { return m_gpuRenderables; }

    private:
        VmaAllocator m_allocator;
        std::vector<GpuRenderable> m_gpuRenderables;
    };
}