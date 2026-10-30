
#pragma once

#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.h>

#include "Scene/Texture.hpp"
#include "VulkanContext.hpp"
#include "Buffer.hpp"

namespace Engine::Renderer
{
    class Texture
    {
    public:
        // Tworzy VkImage, alokuje pamięć przez VMA, tworzy ImageView i Sampler na podstawie Scene::Texture
        Texture(VulkanContext& context, VkCommandPool& commandPool, const Scene::Texture& sourceTexture);
        ~Texture();

        // Gettery do bindowania w Descriptor Set
        VkImageView GetImageView() const { return m_imageView; }
        VkSampler GetSampler() const { return m_sampler; }

    private:
        VmaAllocator m_allocator{VK_NULL_HANDLE};
        VkImage m_image{VK_NULL_HANDLE};
        VmaAllocation m_allocation{VK_NULL_HANDLE};
        VkImageView m_imageView{VK_NULL_HANDLE};
        VkSampler m_sampler{VK_NULL_HANDLE};
        VkDevice m_device{VK_NULL_HANDLE};

        std::unique_ptr<Buffer> createStagingBuffer(const Scene::Texture& sourceTexture);
        
    };
}

// #pragma once

// #include <vulkan/vulkan.h>
// #include <vk_mem_alloc.h>

// #include "Utils/Images/Image.hpp"

// namespace Engine::Renderer
// {
//     class Texture
//     {
//     public:
//         static Texture create(
//             VmaAllocator allocator,
//             VkDevice device,
//             VkCommandPool commandPool,
//             VkQueue graphicsQueue,
//             const Utils::Images::Image &cpuData);

//         void destroy(VmaAllocator allocator, VkDevice device);

//         VkImage getImage() const { return m_image; }
//         VkImageView getImageView() const { return m_imageView; }
//         VkSampler getSampler() const { return m_sampler; }

//     private:
//         VkImage m_image = VK_NULL_HANDLE;
//         VkImageView m_imageView = VK_NULL_HANDLE;
//         VmaAllocation m_allocation = VK_NULL_HANDLE;
//         VkSampler m_sampler = VK_NULL_HANDLE;
//     };
// }
