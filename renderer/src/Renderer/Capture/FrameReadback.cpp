#include "FrameReadback.hpp"

#include <cstring>
#include <stdexcept>

namespace Engine::Renderer
{
    namespace
    {
        constexpr std::array<uint32_t, FrameReadback::TargetCount> kBytesPerPixel = {4, 8, 4, 4, 8};
    }

    FrameReadback::FrameReadback(VulkanContext &context, VkExtent2D extent)
        : m_allocator(context.getAllocator()), m_extent(extent)
    {
        for (size_t i = 0; i < TargetCount; ++i)
        {
            ReadbackBuffer &readback = m_buffers[i];
            readback.size = VkDeviceSize(extent.width) * extent.height * kBytesPerPixel[i];

            VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bufferInfo.size = readback.size;
            bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            VmaAllocationCreateInfo allocInfo{};
            allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
            allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT; // HOST_ACCESS_RANDOM for memory performance when reading

            VmaAllocationInfo allocationInfo{};
            if (vmaCreateBuffer(m_allocator, &bufferInfo, &allocInfo, &readback.buffer, &readback.allocation, &allocationInfo) != VK_SUCCESS)
                throw std::runtime_error("Failed to create readback  buffer");

            readback.mapped = allocationInfo.pMappedData;
        }
    }

    FrameReadback::~FrameReadback()
    {
        for (ReadbackBuffer &readback : m_buffers)
        {
            if (readback.buffer != VK_NULL_HANDLE)
                vmaDestroyBuffer(m_allocator, readback.buffer, readback.allocation);
        }
    }

    void FrameReadback::ReadCopies(VkCommandBuffer commandBuffer, const std::array<Source, TargetCount> &sources)
    {
        std::array<VkImageMemoryBarrier, TargetCount> toTransfer{};
        VkPipelineStageFlags srcStages = 0;

        // Wait for all images to be ready (done writing)
        for (size_t i = 0; i < TargetCount; ++i)
        {
            VkImageMemoryBarrier &barrier = toTransfer[i];
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.srcAccessMask = sources[i].srcAccess;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.oldLayout = sources[i].currentLayout;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = sources[i].image;
            barrier.subresourceRange = {sources[i].aspect, 0, 1, 0, 1};
            srcStages |= sources[i].srcStages;
        }

        vkCmdPipelineBarrier(
            commandBuffer,
            srcStages,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0,
            nullptr,
            0,
            nullptr,
            static_cast<uint32_t>(toTransfer.size()),
            toTransfer.data());

        // Copy images to buffers
        for (size_t i = 0; i < TargetCount; ++i)
        {
            VkBufferImageCopy region{};
            region.imageSubresource = {sources[i].aspect, 0, 0, 1};
            region.imageExtent = {m_extent.width, m_extent.height, 1};

            vkCmdCopyImageToBuffer(
                commandBuffer,
                sources[i].image,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                m_buffers[i].buffer,
                1,
                &region);
        }

        // Expose bufers to host
        VkMemoryBarrier toHost{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        toHost.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toHost.dstAccessMask = VK_ACCESS_HOST_READ_BIT;

        vkCmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT,
            0,
            1,
            &toHost,
            0,
            nullptr,
            0,
            nullptr);
    }

    void FrameReadback::Collect(Capture::FrameData &frame)
    {
        auto copyOut = [this](Target target, auto &destination) {
            const ReadbackBuffer &readback = m_buffers[target];
            vmaInvalidateAllocation(m_allocator, readback.allocation, 0, VK_WHOLE_SIZE);
            destination.resize(readback.size / sizeof(destination[0]));
            std::memcpy(destination.data(), readback.mapped, readback.size);
        };

        frame.width = m_extent.width;
        frame.height = m_extent.height;
        copyOut(Albedo, frame.albedo);
        copyOut(Normal, frame.normals);
        copyOut(Motion, frame.motion);
        copyOut(Depth, frame.depth);
        copyOut(GroundTruth, frame.gt);

    }

} // namespace Engine::Renderer