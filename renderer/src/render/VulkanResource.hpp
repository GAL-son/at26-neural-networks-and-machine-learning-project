#pragma once

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>

#define VK_CHECK(expr)                                                                              \
    do {                                                                                            \
        const VkResult vkCheckResult_ = (expr);                                                     \
        if(vkCheckResult_ != VK_SUCCESS) {                                                          \
            throw std::runtime_error("Vulkan error: " + std::to_string(vkCheckResult_));            \
        }                                                                                           \
    } while(0)                                                                                      \

namespace engine {
    struct Buffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        VmaAllocation allocation = nullptr;
        void* mapped = nullptr;
        VkDeviceSize size = 0;
    };

    struct Image {
        VkImage image = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VmaAllocation allocation = nullptr;
        VkFormat format = VK_FORMAT_UNDEFINED;
        VkExtent2D extent{};
        uint32_t mipLevels = 1;
        VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    };

    enum class MemoryKind {
        GPU_ONLY,
        CPU_TO_GPU,
        GPU_TO_CPU
    };

    Buffer createBuffer(VmaAllocator allocaltor, VkDeviceSize size, VkBufferUsageFlags usage, MemoryKind memoryKind);
    void destroyBuffer(VmaAllocator allocator, Buffer& buffer);

    Image createImage(VkDevice device, VmaAllocator allocator, VkExtent2D extent, VkFormat format, VkImageUsageFlags usage, uint32_t mipLevels = 1);
    void destroyImage(VkDevice device, VmaAllocator allocator, Image& image);

    bool isDepthFormat(VkFormat format);
    uint32_t bytesPerPixel(VkFormat format);

    void transitionImage(VkCommandBuffer commandBuffer, VkImage image, VkImageAspectFlags aspect, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t baseMip = 0, uint32_t mipCount = VK_REMAINING_MIP_LEVELS);
    
    void memoryBarrier(VkCommandBuffer commandBuffer, VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess, VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess);
    void fullBarrier(VkCommandBuffer commandBuffer);

    VkShaderModule loadShaderModule(VkDevice device, const std::filesystem::path& path);
} // namespace engine