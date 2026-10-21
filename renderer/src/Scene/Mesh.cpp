#define GLM_ENABLE_EXPERIMENTAL
#include "Mesh.hpp"
#include <stdexcept>
#include <cstring>
#include <glm/gtx/hash.hpp>

// --- Vertex Input Descriptions for Pipeline Binding ---
VkVertexInputBindingDescription Vertex::getBindingDescription()
{
    VkVertexInputBindingDescription bindingDescription{};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(Vertex);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    return bindingDescription;
}

std::array<VkVertexInputAttributeDescription, 3> Vertex::getAttributeDescriptions()
{
    std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};

    // Position (Location 0)
    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[0].offset = offsetof(Vertex, position);

    // Normal (Location 1)
    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[1].offset = offsetof(Vertex, normal);

    // Texture Coordinates (Location 2)
    attributeDescriptions[2].binding = 0;
    attributeDescriptions[2].location = 2;
    attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

    return attributeDescriptions;
}

// --- Mesh Implementation ---

Mesh::Mesh(VmaAllocator allocator, const std::vector<Vertex> &vertices, const std::vector<uint32_t> &indices)
    : m_allocator(allocator), m_vertexCount(static_cast<uint32_t>(vertices.size())), m_indexCount(static_cast<uint32_t>(indices.size()))
{
    // 1. Create and populate Vertex Buffer
    VkDeviceSize vertexBufferSize = sizeof(Vertex) * vertices.size();
    createBuffer(vertexBufferSize,
                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VMA_MEMORY_USAGE_AUTO,
                 m_vertexBuffer, m_vertexAllocation,
                 vertices.data());

    // 2. Create and populate Index Buffer
    VkDeviceSize indexBufferSize = sizeof(uint32_t) * indices.size();
    createBuffer(indexBufferSize,
                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VMA_MEMORY_USAGE_AUTO,
                 m_indexBuffer, m_indexAllocation,
                 indices.data());
}

Mesh::~Mesh()
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

namespace std
{
    template <>
    struct hash<Vertex>
    {
        size_t operator()(Vertex const &vertex) const
        {
            return ((hash<glm::vec3>()(vertex.position) ^ (hash<glm::vec3>()(vertex.normal) << 1)) >> 1) ^ (hash<glm::vec2>()(vertex.texCoord) << 1);
        }
    };
}

std::unique_ptr<Mesh> Mesh::createFromLoadedData(VmaAllocator allocator, const Utils::LoadedMeshData &rawData)
{
    size_t vertexCount = rawData.positions.size() / 3;
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    vertices.reserve(vertexCount);
    indices.reserve(vertexCount);

    for (size_t i = 0; i < vertexCount; ++i)
    {
        Vertex v{};
        v.position = {rawData.positions[i * 3 + 0], rawData.positions[i * 3 + 1], rawData.positions[i * 3 + 2]};
        
        if (i * 3 + 2 < rawData.normals.size())
        {
            v.normal = {rawData.normals[i * 3 + 0], rawData.normals[i * 3 + 1], rawData.normals[i * 3 + 2]};
        }
        
        if (i * 2 + 1 < rawData.texCoords.size())
        {
            v.texCoord = {rawData.texCoords[i * 2 + 0], rawData.texCoords[i * 2 + 1]};
        }

        vertices.push_back(v);
        indices.push_back(static_cast<uint32_t>(i));
    }

    glm::vec3 minBox(FLT_MAX);
    glm::vec3 maxBox(-FLT_MAX);
    for (const auto& v : vertices) {
        minBox = glm::min(minBox, v.position);
        maxBox = glm::max(maxBox, v.position);
    }
    std::cout << "--- MODEL BOUNDING BOX ---\n";
    std::cout << "Min: X=" << minBox.x << ", Y=" << minBox.y << ", Z=" << minBox.z << "\n";
    std::cout << "Max: X=" << maxBox.x << ", Y=" << maxBox.y << ", Z=" << maxBox.z << "\n";
    std::cout << "--------------------------\n";

    return std::make_unique<Mesh>(allocator, vertices, indices);
}

void Mesh::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage,
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
        throw std::runtime_error("Failed to create VMA buffer for mesh!");
    }

    if (data != nullptr)
    {
        void *mappedData;
        vmaMapMemory(m_allocator, outAllocation, &mappedData);
        std::memcpy(mappedData, data, static_cast<size_t>(size));
        vmaUnmapMemory(m_allocator, outAllocation);
    }
}

void Mesh::draw(VkCommandBuffer commandBuffer) const {
    VkBuffer vertexBuffers[] = {m_vertexBuffer};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
    vkCmdBindIndexBuffer(commandBuffer, m_indexBuffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(commandBuffer, m_indexCount, 1, 0, 0, 0);
}