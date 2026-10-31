#pragma once

#include <memory>
#include <vector>
#include <vulkan/vulkan.hpp>

#include "VulkanContext.hpp"
#include "Core/Window.hpp"
#include "Core/Camera.hpp"
#include "Renderer/Resources/Scene.hpp"
#include "Renderer/Passes/GeometryPass.hpp"
#include "Renderer/Passes/CompositePass.hpp"
#include "UI/UIManager.hpp"

namespace Engine::Renderer
{
    class VulkanRenderer
    {
    public:
        VulkanRenderer(VulkanContext& vulkanContext, const Window& window);
        ~VulkanRenderer();

        void init();
        void cleanup();

        // Zarządzanie sceną GPU
        void loadScene(const Engine::Scene::Scene& cpuScene);

        // Główna pętla renderowania
        void renderFrame(Engine::Camera& camera, uint32_t processedFrames, float frameTimeMs, float passTimeMs);

    private:
        VulkanContext& m_vulkanContext;
        const Window& m_window;

        // Command Pool dla alokacji komend renderowania i uploadu
        VkCommandPool m_commandPool{VK_NULL_HANDLE};
        VkCommandBuffer m_commandBuffer{VK_NULL_HANDLE};

        // Sampler wykorzystywany do czytania buforów MRT w composite passie
        VkSampler m_defaultSampler{VK_NULL_HANDLE};

        // Pasy renderowania
        std::unique_ptr<GeometryPass> m_geometryPass;
        std::unique_ptr<CompositePass> m_compositePass;

        // Zarządcza scena GPU
        std::unique_ptr<Renderer::Scene> m_activeGpuScene;

        // UI Manager (ImGui)
        UIManager m_uiManager;
        VkDescriptorPool m_imguiPool{VK_NULL_HANDLE};

        // Synchronizacja
        VkSemaphore m_imageAvailableSemaphore{VK_NULL_HANDLE};
        std::vector<VkSemaphore> m_renderFinishedSemaphores;
        VkFence m_inFlightFence{VK_NULL_HANDLE};

        void createCommandPool();
        void createCommandBuffers();
        void createSampler();
        void createSyncPrimitives();

        void waitForInFlightFence();
        uint32_t acquireNextSwapchainImage();
        void beginCommandBuffer();
        
        void prepareSwapchainForRendering(uint32_t imageIndex);
        void prepareSwapchainForPresentation(uint32_t imageIndex);
        void insertImageMemoryBarrier(
            VkImage image,
            VkImageLayout oldLayout,
            VkImageLayout newLayout,
            VkAccessFlags srcAccess,
            VkAccessFlags dstAccess,
            VkPipelineStageFlags srcStage,
            VkPipelineStageFlags dstStage
        );

        void submitCommandBuffer(uint32_t imageIndex);
        void presentSwapchainImage(uint32_t imageIndex);
    };
}