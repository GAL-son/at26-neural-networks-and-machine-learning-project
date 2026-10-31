#include "Buffer.hpp"
#include <stdexcept>

namespace Engine::Renderer
{
    Buffer::Buffer(VmaAllocator allocator, VkDeviceSize size, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage)
        : m_allocator(allocator), m_size(size)
    {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = memoryUsage;
        allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

        if (vmaCreateBuffer(m_allocator, &bufferInfo, &allocInfo, &m_buffer, &m_allocation, nullptr) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create and allocate Buffer using VMA!");
        }
    }

    Buffer::~Buffer()
    {
        if (m_buffer != VK_NULL_HANDLE && m_allocation != VK_NULL_HANDLE)
        {
            // VMA bezpiecznie zwalnia bufor i pamięć
            vmaDestroyBuffer(m_allocator, m_buffer, m_allocation);
        }
    }

    void Buffer::UploadData(const void* data)
    {
        void* mappedData = nullptr;
        
        vmaMapMemory(m_allocator, m_allocation, &mappedData);
        std::memcpy(mappedData, data, m_size);
        vmaUnmapMemory(m_allocator, m_allocation);
    }
}