#pragma one

#define VULKAN_HPP_NO_CONSTRUCTORS

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <vector>
#include "../../Scene/Mesh.hpp"

namespace Engine::Renderer
{

    class VulkanMesh
    {
    public:
        VulkanMesh(VmaAllocator allocator, const Engine::Scene::Mesh &meshData);
        ~VulkanMesh();

        // Prevent copying
        VulkanMesh(const VulkanMesh &) = delete;
        VulkanMesh &operator=(const VulkanMesh &) = delete;

        // Allow moving
        VulkanMesh(VulkanMesh &&) noexcept = default;
        VulkanMesh &operator=(VulkanMesh &&) noexcept = default;

        void draw(VkCommandBuffer commandBuffer, const std::vector<Engine::Scene::Submesh> &submeshes) const;

    private:
        VmaAllocator m_allocator;

        VkBuffer m_vertexBuffer = VK_NULL_HANDLE;
        VmaAllocation m_vertexAllocation = VK_NULL_HANDLE;

        VkBuffer m_indexBuffer = VK_NULL_HANDLE;
        VmaAllocation m_indexAllocation = VK_NULL_HANDLE;
        uint32_t m_indexCount = 0;

        void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage,
                          VkBuffer &outBuffer, VmaAllocation &outAllocation, const void *data);
    };

} // namespace Engine::Renderer