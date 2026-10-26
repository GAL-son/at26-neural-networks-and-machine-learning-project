#include "VulkanMesh.hpp"
#include <stdexcept>
#include <cstring>

namespace Engine::Renderer
{

    VulkanMesh::VulkanMesh(VmaAllocator allocator, const Engine::Scene::Mesh &meshData)
        : m_allocator(allocator), m_indexCount(static_cast<uint32_t>(meshData.indices.size()))
    {
        // 1. Create Vertex Buffer
        VkDeviceSize vertexBufferSize = sizeof(Engine::Scene::Vertex) * meshData.vertices.size();
        createBuffer(vertexBufferSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO,
                     m_vertexBuffer, m_vertexAllocation, meshData.vertices.data());

        // 2. Create Index Buffer
        VkDeviceSize indexBufferSize = sizeof(uint32_t) * meshData.indices.size();
        createBuffer(indexBufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO,
                     m_indexBuffer, m_indexAllocation, meshData.indices.data());
    }

    void VulkanMesh::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage,
                                  VkBuffer &outBuffer, VmaAllocation &outAllocation, const void *data)
    {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = memoryUsage;
        allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

        if (vmaCreateBuffer(m_allocator, &bufferInfo, &allocInfo, &outBuffer, &outAllocation, nullptr) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create VMA buffer for VulkanMesh!");
        }

        if (data != nullptr)
        {
            void *mappedData;
            vmaMapMemory(m_allocator, outAllocation, &mappedData);
            std::memcpy(mappedData, data, static_cast<size_t>(size));
            vmaUnmapMemory(m_allocator, outAllocation);
        }
    }

    void VulkanMesh::draw(VkCommandBuffer commandBuffer, const std::vector<Engine::Scene::Submesh> &submeshes) const
    {
        VkDeviceSize offsets[] = {0};
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, &m_vertexBuffer, offsets);
        vkCmdBindIndexBuffer(commandBuffer, m_indexBuffer, 0, VK_INDEX_TYPE_UINT32);

        if (!submeshes.empty())
        {
            for (const auto &submesh : submeshes)
            {
                vkCmdDrawIndexed(commandBuffer, submesh.indexCount, 1, submesh.firstIndex, 0, 0);
            }
        }
        else
        {
            vkCmdDrawIndexed(commandBuffer, m_indexCount, 1, 0, 0, 0);
        }
    }

    VulkanMesh::~VulkanMesh()
    {
        if (m_vertexBuffer != VK_NULL_HANDLE && m_vertexAllocation != VK_NULL_HANDLE)
        {
            vmaDestroyBuffer(m_allocator, m_vertexBuffer, m_vertexAllocation);
        }
        if (m_indexBuffer != VK_NULL_HANDLE && m_indexAllocation != VK_NULL_HANDLE)
        {
            vmaDestroyBuffer(m_allocator, m_indexBuffer, m_indexAllocation);
        }
    }

} // namespace Engine::Renderer