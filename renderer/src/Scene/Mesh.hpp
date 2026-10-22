#pragma once

#define VULKAN_HPP_NO_CONSTRUCTORS

#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.h>
#include <glm/glm.hpp>
#include <vector>
#include <array>
#include <memory>

#include "../Utils/FileUtils.hpp"

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texCoord;

    static VkVertexInputBindingDescription getBindingDescription();
    static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions();
};

class Mesh {
public:
    Mesh(VmaAllocator allocator, const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices);
    ~Mesh();

    static std::unique_ptr<Mesh> createFromLoadedData(VmaAllocator allocator, const Utils::LoadedMeshData& rawData);

    // Prevent copying to avoid double-free of VMA allocations
    Mesh(const Mesh&) = delete;
    Mesh& operator=(const Mesh&) = delete;

    void draw(VkCommandBuffer commandBuffer) const;

    uint32_t getVertexCount() const { return m_vertexCount; }
    uint32_t getIndexCount() const { return m_indexCount; }

private:
    VmaAllocator m_allocator;
    
    VkBuffer m_vertexBuffer = VK_NULL_HANDLE;
    VmaAllocation m_vertexAllocation = VK_NULL_HANDLE;
    uint32_t m_vertexCount = 0;

    VkBuffer m_indexBuffer = VK_NULL_HANDLE;
    VmaAllocation m_indexAllocation = VK_NULL_HANDLE;
    uint32_t m_indexCount = 0;

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage, 
                      VkBuffer& outBuffer, VmaAllocation& outAllocation, const void* data = nullptr);
};