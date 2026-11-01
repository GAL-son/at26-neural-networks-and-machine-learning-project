#pragma once
#define VULKAN_HPP_NO_CONSTRUCTORS
#include <vulkan/vulkan.hpp>
#include "../Core/Window.hpp"
#include "../Renderer/VulkanContext.hpp"

namespace Engine::UI
{

    class UIManager
    {
    public:
        UIManager();
        ~UIManager();

        void init(const Window &window, const VulkanContext &vulkanContext, VkDescriptorPool &outDescriptorPool);
        void shutdown(const VulkanContext &vulkanContext, VkDescriptorPool descriptorPool);

        void beginFrame();
        void renderStats(uint32_t processedFrames, float frameTimeMs, float passTimeMs);
        void renderBufferLabels();
        void renderDrawData(VkCommandBuffer commandBuffer);

    private:
        bool m_initialized = false;
        VkFormat m_swapchainFormat;
    };
}