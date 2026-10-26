#pragma once

#include <vector>
#include <string>
#include <glm/glm.hpp>

namespace Engine::Scene
{

    struct Vertex
    {
        glm::vec3 position;
        glm::vec3 normal;
        glm::vec2 texCoord;
    };

    struct Submesh
    {
        uint32_t indexCount = 0;
        uint32_t firstIndex = 0;
        int materialIndex = -1;
    };

    struct Mesh
    {
        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;
        std::vector<Submesh> submeshes;
    };

    struct Material
    {
        std::string diffuseTexPath;
        glm::vec3 diffuseColor{1.0f};
        float roughness = 0.5f;
    };

} // namespace Engine::Scene

// #pragma once

// #define VULKAN_HPP_NO_CONSTRUCTORS

// #include <vulkan/vulkan.h>
// #include <vk_mem_alloc.h>
// #include <glm/glm.hpp>
// #include <vector>
// #include <array>
// #include <memory>

// #include "../Utils/FileUtils.hpp"

// struct Vertex {
//     glm::vec3 position;
//     glm::vec3 normal;
//     glm::vec2 texCoord;

//     static VkVertexInputBindingDescription getBindingDescription();
//     static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions();
// };

// // Submesh structure pointing to an index range and a material index
// struct Submesh {
//     uint32_t indexCount = 0;
//     uint32_t firstIndex = 0;
//     int materialIndex = -1;
// };

// class Mesh {
// public:
//     Mesh(VmaAllocator allocator, const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices, const std::vector<Submesh>& submeshes);
//     ~Mesh();

//     static std::unique_ptr<Mesh> createFromLoadedData(VmaAllocator allocator, const Utils::LoadedMeshData& rawData);

//     // Prevent copying
//     Mesh(const Mesh&) = delete;
//     Mesh& operator=(const Mesh&) = delete;

//     void draw(VkCommandBuffer commandBuffer) const;

//     uint32_t getVertexCount() const { return m_vertexCount; }
//     uint32_t getIndexCount() const { return m_indexCount; }
//     const std::vector<Submesh>& getSubmeshes() const { return m_submeshes; }

// private:
//     VmaAllocator m_allocator;
    
//     VkBuffer m_vertexBuffer = VK_NULL_HANDLE;
//     VmaAllocation m_vertexAllocation = VK_NULL_HANDLE;
//     uint32_t m_vertexCount = 0;

//     VkBuffer m_indexBuffer = VK_NULL_HANDLE;
//     VmaAllocation m_indexAllocation = VK_NULL_HANDLE;
//     uint32_t m_indexCount = 0;

//     std::vector<Submesh> m_submeshes;

//     void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage, 
//                       VkBuffer& outBuffer, VmaAllocation& outAllocation, const void* data = nullptr);
// };