#include "VulkanResource.hpp"

#include <fstream>
#include <vector>

namespace engine {
    Buffer createBuffer(VmaAllocator allocator, VkDeviceSize size, VkBufferUsageFlags usage, MemoryKind memoryKind) {
        VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        // Select memory usage based on the specified memory kind
        VmaAllocationCreateInfo allocInfo{};
        switch(memoryKind) {
            case MemoryKind::GPU_ONLY:
                // allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY; // Use VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE for better performance
                allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
                break;
            case MemoryKind::CPU_TO_GPU:
                // allocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU; // Use VMA_MEMORY_USAGE_AUTO for better performance
                allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
                allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
                break;
            case MemoryKind::GPU_TO_CPU:
                // allocInfo.usage = VMA_MEMORY_USAGE_GPU_TO_CPU; // Use VMA_MEMORY_USAGE_AUTO for better performance
                allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
                allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
                break;
        }

        Buffer buffer;
        VmaAllocationInfo info{};
        VK_CHECK(vmaCreateBuffer(allocator, &bufferInfo, &allocInfo, &buffer.buffer, &buffer.allocation, &info));

        // if(memoryKind == MemoryKind::CPU_TO_GPU || memoryKind == MemoryKind::GPU_TO_CPU) {
        //     VK_CHECK(vmaMapMemory(allocator, buffer.allocation, &buffer.mapped));
        // }

        buffer.mapped = info.pMappedData;
        buffer.size = size;
        return buffer;
    }

    void destroyBuffer(VmaAllocator allocator, Buffer& buffer) {
        if(buffer.buffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(allocator, buffer.buffer, buffer.allocation);
            buffer = {};
        }
    }

    bool isDepthFormat(VkFormat format) {
        switch(format) {
            case VK_FORMAT_D16_UNORM:
            case VK_FORMAT_D32_SFLOAT:
            case VK_FORMAT_D24_UNORM_S8_UINT:
            case VK_FORMAT_D32_SFLOAT_S8_UINT:
                return true;
            default:
                return false;
        }
    }

    uint32_t bytesPerPixel(VkFormat format) {
        switch(format) {
            case VK_FORMAT_R8_UNORM:
                return 1;
            case VK_FORMAT_R8G8_UNORM:
                return 2;
            case VK_FORMAT_R8G8B8A8_UNORM:
            case VK_FORMAT_R8G8B8A8_SRGB:
            case VK_FORMAT_B8G8R8A8_UNORM:
            case VK_FORMAT_R16G16_SFLOAT:
            case VK_FORMAT_R32_SFLOAT:
            case VK_FORMAT_D32_SFLOAT:
                return 4;
            case VK_FORMAT_R16G16B16A16_SFLOAT:
                return 8;
            case VK_FORMAT_R32G32B32A32_SFLOAT:
                return 16;
            default:
                throw std::runtime_error("Unsupported format for bytesPerPixel");
        }
    }

    Image createImage(VkDevice device, VmaAllocator allocator, VkExtent2D extent, VkFormat format, VkImageUsageFlags usage, uint32_t mipLevels) {
        Image image;
        image.format = format;
        image.extent = extent;
        image.mipLevels = mipLevels;
        image.aspect = isDepthFormat(format) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;

        VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent = {extent.width, extent.height, 1};
        imageInfo.mipLevels = mipLevels;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage = usage;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        VK_CHECK(vmaCreateImage(allocator, &imageInfo, &allocInfo, &image.image, &image.allocation, nullptr));

        VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        viewInfo.image = image.image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange = {image.aspect, 0, mipLevels, 0, 1};
        VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &image.view));
        
        return image;
    }

    void destroyImage(VkDevice device, VmaAllocator allocator, Image& image) {
        if(image.view != VK_NULL_HANDLE) {
            vkDestroyImageView(device, image.view, nullptr);
        }
        if(image.image != VK_NULL_HANDLE) {
            vmaDestroyImage(allocator, image.image, image.allocation);
        }

        image = {};
    }

    void MemoryBarrier(VkCommandBuffer commandBuffer, VkImage image, VkImageAspectFlags aspect, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t baseMip, uint32_t mipCount) {
        VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2}
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    }


}