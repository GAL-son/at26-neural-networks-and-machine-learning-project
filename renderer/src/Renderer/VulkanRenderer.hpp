#pragma once

#include "../Utils/Macros.hpp"

#define VULKAN_HPP_NO_CONSTRUCTORS
#include <vulkan/vulkan.h>
#include <memory>
#include <vector>

#include "VulkanContext.hpp"
#include "Passes/GeometryPass.hpp"
#include "Passes/CompositePass.hpp"
#include "../UI/UIManager.hpp"
#include "../Core/Window.hpp"
#include "../Scene/Scene.hpp"

namespace Engine::Renderer
{
    /**
     * @brief Manages the main Vulkan rendering loop, synchronization, and pass coordination.
     */
    class VulkanRenderer
    {
    public:
        VulkanRenderer(VulkanContext &vulkanContext, const Window &window);
        ~VulkanRenderer();

        DISALLOW_COPY(VulkanRenderer)

        /** @brief Initializes command pools, synchronization primitives, and rendering passes. */
        void init();

        /** @brief Safely releases all managed Vulkan resources. */
        void cleanup();

        /**
         * @brief Executes a single frame rendering and UI composition cycle.
         * @note Blocks on the in-flight fence until the previous frame completes.
         */
        void renderFrame(Scene::Scene *scene, uint32_t processedFrames, float frameTimeMs, float passTimeMs);

    private:
        VulkanContext &m_vulkanContext;
        const Window &m_window;

        // Vulkan synchronization & command primitives
        VkCommandPool m_commandPool = VK_NULL_HANDLE;
        VkCommandBuffer m_commandBuffer = VK_NULL_HANDLE;
        VkSemaphore m_imageAvailableSemaphore = VK_NULL_HANDLE;
        std::vector<VkSemaphore> m_renderFinishedSemaphores;
        VkFence m_inFlightFence = VK_NULL_HANDLE;
        VkDescriptorPool m_imguiPool = VK_NULL_HANDLE;

        // Passes & UI
        GeometryPass m_geometryPass;
        CompositePass m_compositePass;
        UIManager m_uiManager;

        void createCommandPool();
        void createCommandBuffers();
        void createSyncPrimitives();
        void waitForInFlightFence();
        uint32_t acquireNextSwapchainImage();
        void beginCommandBuffer();
        void recordGeometryAndClears(Scene::Scene *scene);
        void prepareSwapchainForRendering(uint32_t imageIndex);
        void prepareSwapchainForPresentation(uint32_t imageIndex);
        void insertImageMemoryBarrier(
            VkImage image,
            VkImageLayout oldLayout,
            VkImageLayout newLayout,
            VkAccessFlags srcAccess,
            VkAccessFlags dstAccess,
            VkPipelineStageFlags srcStage,
            VkPipelineStageFlags dstStage);

        void recordUIAndComposition(uint32_t imageIndex, uint32_t processedFrames, float frameTimeMs, float passTimeMs);
        void submitCommandBuffer(uint32_t imageIndex);
        void presentSwapchainImage(uint32_t imageIndex);

    };
}