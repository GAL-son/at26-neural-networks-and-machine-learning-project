#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "Utils/Images/Image.hpp"

namespace Engine::Renderer
{
    class Texture
    {
    public:
        static Texture create(
            VmaAllocator allocator,
            VkDevice device,
            VkCommandPool commandPool,
            VkQueue graphicsQueue,
            const Utils::Images::Image &cpuData);

        void destroy(VmaAllocator allocator, VkDevice device);

        VkImage getImage() const { return m_image; }
        VkImageView getImageView() const { return m_imageView; }
        VkSampler getSampler() const { return m_sampler; }

    private:
        VkImage m_image = VK_NULL_HANDLE;
        VkImageView m_imageView = VK_NULL_HANDLE;
        VmaAllocation m_allocation = VK_NULL_HANDLE;
        VkSampler m_sampler = VK_NULL_HANDLE;
    };
}
