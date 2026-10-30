#pragma once

#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.h>
#include "Utils/Macros.hpp"

namespace Engine::Renderer
{
    class Buffer
    {
    public:
        Buffer(VmaAllocator allocator, VkDeviceSize size, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);
        ~Buffer();

        DISALLOW_COPY(Buffer)
        ALLOW_COPY(Buffer)

        void UploadData(const void* data);

        VkBuffer GetHandle() const { return m_buffer; }

    private:
        VmaAllocator m_allocator{VK_NULL_HANDLE};
        VkBuffer m_buffer{VK_NULL_HANDLE};
        VmaAllocation m_allocation{VK_NULL_HANDLE};
        VkDeviceSize m_size{0};
    };
}