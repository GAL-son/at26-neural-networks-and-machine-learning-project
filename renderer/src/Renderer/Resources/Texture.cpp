#include "Renderer/Resources/Texture.hpp"
#include <stdexcept>
#include <cstring>


namespace Engine::Renderer
{
    Texture Texture::create(
        VmaAllocator allocator,
        VkDevice device,
        VkCommandPool commandPool,
        VkQueue graphicsQueue,
        const Utils::Images::Image &cpuData)
    {
        Texture texture{};
        VkDeviceSize imageSize = cpuData.width * cpuData.height * 4;

        // 1. Create Staging Buffer via VMA
        VkBuffer stagingBuffer;
        VmaAllocation stagingAllocation;

        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = imageSize;
        bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
        allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

        vmaCreateBuffer(allocator, &bufferInfo, &allocInfo, &stagingBuffer, &stagingAllocation, nullptr);

        void *data;
        vmaMapMemory(allocator, stagingAllocation, &data);
        memcpy(data, cpuData.pixels.data(), static_cast<size_t>(imageSize));
        vmaUnmapMemory(allocator, stagingAllocation);

        // 2. Create GPU Image via VMA
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = static_cast<uint32_t>(cpuData.width);
        imageInfo.extent.height = static_cast<uint32_t>(cpuData.height);
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo imageAllocInfo{};
        imageAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;

        vmaCreateImage(allocator, &imageInfo, &imageAllocInfo, &texture.m_image, &texture.m_allocation, nullptr);

        // 3. Transition layout, copy buffer to image, transition to shader read
        // (Using your single-time command helpers...)
        // transitionImageLayout(...)
        // copyBufferToImage(...)
        // transitionImageLayout(...)

        vmaDestroyBuffer(allocator, stagingBuffer, stagingAllocation);

        // 4. Create ImageView and Sampler
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = texture.m_image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        vkCreateImageView(device, &viewInfo, nullptr, &texture.m_imageView);

        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.anisotropyEnable = VK_TRUE;
        samplerInfo.maxAnisotropy = 16.0f;

        vkCreateSampler(device, &samplerInfo, nullptr, &texture.m_sampler);

        return texture;
    }

    void Texture::destroy(VmaAllocator allocator, VkDevice device)
    {
        if (m_sampler)
            vkDestroySampler(device, m_sampler, nullptr);
        if (m_imageView)
            vkDestroyImageView(device, m_imageView, nullptr);
        if (m_image)
            vmaDestroyImage(allocator, m_image, m_allocation);
    }
}