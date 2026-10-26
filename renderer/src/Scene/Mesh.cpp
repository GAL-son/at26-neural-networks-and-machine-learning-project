// #define GLM_ENABLE_EXPERIMENTAL
// #include "Mesh.hpp"
// #include <stdexcept>
// #include <cstring>
// #include <glm/gtx/hash.hpp>

// // --- Vertex Input Descriptions for Pipeline Binding ---
// VkVertexInputBindingDescription Vertex::getBindingDescription()
// {
//     VkVertexInputBindingDescription bindingDescription{};
//     bindingDescription.binding = 0;
//     bindingDescription.stride = sizeof(Vertex);
//     bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
//     return bindingDescription;
// }

// std::array<VkVertexInputAttributeDescription, 3> Vertex::getAttributeDescriptions()
// {
//     std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};

//     // Position (Location 0)
//     attributeDescriptions[0].binding = 0;
//     attributeDescriptions[0].location = 0;
//     attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
//     attributeDescriptions[0].offset = offsetof(Vertex, position);

//     // Normal (Location 1)
//     attributeDescriptions[1].binding = 0;
//     attributeDescriptions[1].location = 1;
//     attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
//     attributeDescriptions[1].offset = offsetof(Vertex, normal);

//     // Texture Coordinates (Location 2)
//     attributeDescriptions[2].binding = 0;
//     attributeDescriptions[2].location = 2;
//     attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
//     attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

//     return attributeDescriptions;
// }

// // --- Mesh Implementation ---
// Mesh::Mesh(VmaAllocator allocator, const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices, const std::vector<Submesh>& submeshes)
//     : m_allocator(allocator), m_vertexCount(static_cast<uint32_t>(vertices.size())), 
//       m_indexCount(static_cast<uint32_t>(indices.size())), m_submeshes(submeshes)
// {
//     // 1. Create Vertex Buffer
//     VkDeviceSize vertexBufferSize = sizeof(Vertex) * vertices.size();
//     createBuffer(vertexBufferSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO, 
//                  m_vertexBuffer, m_vertexAllocation, vertices.data());

//     // 2. Create Index Buffer
//     VkDeviceSize indexBufferSize = sizeof(uint32_t) * indices.size();
//     createBuffer(indexBufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO, 
//                  m_indexBuffer, m_indexAllocation, indices.data());
// }

// std::unique_ptr<Mesh> Mesh::createFromLoadedData(VmaAllocator allocator, const Utils::LoadedMeshData& rawData)
// {
//     std::vector<Vertex> vertices;
//     vertices.reserve(rawData.positions.size() / 3);

//     for (size_t i = 0; i < rawData.positions.size() / 3; ++i) {
//         Vertex v{};
//         v.position = { rawData.positions[3 * i + 0], rawData.positions[3 * i + 1], rawData.positions[3 * i + 2] };
//         v.normal   = { rawData.normals[3 * i + 0], rawData.normals[3 * i + 1], rawData.normals[3 * i + 2] };
//         v.texCoord = { rawData.texCoords[2 * i + 0], rawData.texCoords[2 * i + 1] };
//         vertices.push_back(v);
//     }

//     // Convert raw submeshes to our Mesh submesh structure
//     std::vector<Submesh> submeshes;
//     for (const auto& rawSub : rawData.submeshes) {
//         Submesh sm{};
//         sm.indexCount = rawSub.indexCount;
//         sm.firstIndex = rawSub.firstIndex;
//         sm.materialIndex = rawSub.materialIndex;
//         submeshes.push_back(sm);
//     }

//     return std::make_unique<Mesh>(allocator, vertices, rawData.indices, submeshes);
// }

// void Mesh::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage,
//                         VkBuffer &outBuffer, VmaAllocation &outAllocation, const void *data)
// {
//     VkBufferCreateInfo bufferInfo{};
//     bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
//     bufferInfo.size = size;
//     bufferInfo.usage = usage;
//     bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

//     VmaAllocationCreateInfo allocInfo{};
//     allocInfo.usage = memoryUsage;
//     allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

//     if (vmaCreateBuffer(m_allocator, &bufferInfo, &allocInfo, &outBuffer, &outAllocation, nullptr) != VK_SUCCESS)
//     {
//         throw std::runtime_error("Failed to create VMA buffer for mesh!");
//     }

//     if (data != nullptr)
//     {
//         void *mappedData;
//         vmaMapMemory(m_allocator, outAllocation, &mappedData);
//         std::memcpy(mappedData, data, static_cast<size_t>(size));
//         vmaUnmapMemory(m_allocator, outAllocation);
//     }
// }

// void Mesh::draw(VkCommandBuffer commandBuffer) const {
//     VkDeviceSize offsets[] = {0};
//     vkCmdBindVertexBuffers(commandBuffer, 0, 1, &m_vertexBuffer, offsets);
//     vkCmdBindIndexBuffer(commandBuffer, m_indexBuffer, 0, VK_INDEX_TYPE_UINT32);

//     // If submeshes exist, draw by submesh range. Otherwise, draw the whole index count.
//     if (!m_submeshes.empty()) {
//         for (const auto& submesh : m_submeshes) {
//             // Later, we will bind material descriptor sets right here before this draw call!
//             vkCmdDrawIndexed(commandBuffer, submesh.indexCount, 1, submesh.firstIndex, 0, 0);
//         }
//     } else {
//         vkCmdDrawIndexed(commandBuffer, m_indexCount, 1, 0, 0, 0);
//     }
// }

// Mesh::~Mesh()
// {
//     if (m_vertexBuffer != VK_NULL_HANDLE && m_vertexAllocation != VK_NULL_HANDLE)
//     {
//         vmaDestroyBuffer(m_allocator, m_vertexBuffer, m_vertexAllocation);
//     }
//     if (m_indexBuffer != VK_NULL_HANDLE && m_indexAllocation != VK_NULL_HANDLE)
//     {
//         vmaDestroyBuffer(m_allocator, m_indexBuffer, m_indexAllocation);
//     }
// }

// namespace std
// {
//     template <>
//     struct hash<Vertex>
//     {
//         size_t operator()(Vertex const &vertex) const
//         {
//             return ((hash<glm::vec3>()(vertex.position) ^ (hash<glm::vec3>()(vertex.normal) << 1)) >> 1) ^ (hash<glm::vec2>()(vertex.texCoord) << 1);
//         }
//     };
// }


