#pragma once
#include "Utils/Macros.hpp"

#include <array>
#include <vk_mem_alloc.h>

#include "Renderer/VulkanContext.hpp"
#include "FrameData.hpp"

namespace Engine::Renderer
{
    class FrameReadback
    {
    public:
        enum Target
        {
            Albedo,
            Normal,
            Motion,
            Depth,
            GroundTruth,
            TargetCount
        };
        struct Source
        {
            VkImage image;
            VkImageLayout currentLayout;
            VkPipelineStageFlags srcStages;
            VkAccessFlags srcAccess;
            VkImageAspectFlags aspect;
        };

        FrameReadback(VulkanContext &context, VkExtent2D extent);
        ~FrameReadback();

        DISALLOW_COPY(FrameReadback);

        void ReadCopies(VkCommandBuffer commandBuffer, const std::array<Source, TargetCount> &sources);
        void Collect(Capture::FrameData &frame);

    private:
        struct ReadbackBuffer
        {
            VkBuffer buffer{VK_NULL_HANDLE};
            VmaAllocation allocation{VK_NULL_HANDLE};
            void *mapped{nullptr};
            VkDeviceSize size{0};
        };

        VmaAllocator m_allocator;
        VkExtent2D m_extent;
        std::array<ReadbackBuffer, TargetCount> m_buffers{};
    };
}