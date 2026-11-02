#pragma once

#include <memory>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <optional>

#include "VulkanContext.hpp"
#include "Core/Window.hpp"
#include "Core/Camera.hpp"
#include "Renderer/Resources/Scene.hpp"
#include "Renderer/Passes/GeometryPass.hpp"
#include "Renderer/Passes/CompositePass.hpp"
#include "Renderer/Passes/GroundTruthPass.hpp"
#include "Renderer/Capture/FrameReadback.hpp"
#include "Scene/Scene.hpp"

struct ImDrawData;

namespace Engine::Renderer
{
    class VulkanRenderer
    {
    public:
        VulkanRenderer(VulkanContext &vulkanContext, VkExtent2D renderExtent);
        ~VulkanRenderer();

        void init();
        void cleanup();

        // Zarządzanie sceną GPU
        // void loadScene(const Engine::Scene::Scene &cpuScene);
        std::unique_ptr<Renderer::Scene> createGpuScene(const ::Engine::Scene::Scene &cpuScene) const;
        void SetScene(std::unique_ptr<Renderer::Scene> scene);
        bool hasScene() const { return m_activeGpuScene != nullptr; };

        // Główna pętla renderowania
        std::optional<Capture::FrameData> renderFrame(
            ::Engine::Camera &camera,
            bool capture = false,
            ImDrawData *uiDrawData = nullptr
        );

    private:
        static constexpr uint32_t kSsaaFactor = 4; // 4x4 = SSAA x16
        bool m_renderGroundTruth = true;

        VulkanContext &m_vulkanContext;
        VkExtent2D m_renderExtent;

        // Command Pool dla alokacji komend renderowania i uploadu
        VkCommandPool m_commandPool{VK_NULL_HANDLE};
        VkCommandBuffer m_commandBuffer{VK_NULL_HANDLE};

        // Sampler wykorzystywany do czytania buforów MRT w composite passie
        VkSampler m_defaultSampler{VK_NULL_HANDLE};

        // Pasy renderowania
        std::unique_ptr<GeometryPass> m_geometryPass;
        std::unique_ptr<CompositePass> m_compositePass;
        std::unique_ptr<GroundTruthPass> m_groundTruthPass;

        // Zarządcza scena GPU
        std::unique_ptr<Renderer::Scene> m_activeGpuScene;
        VkDescriptorSetLayout m_materialSetLayout{VK_NULL_HANDLE};

        // Synchronizacja
        VkSemaphore m_imageAvailableSemaphore{VK_NULL_HANDLE};
        std::vector<VkSemaphore> m_renderFinishedSemaphores;
        VkFence m_inFlightFence{VK_NULL_HANDLE};

        std::unique_ptr<FrameReadback> m_frameReadback;
        std::array<FrameReadback::Source, FrameReadback::TargetCount> readbackSources();

        void createCommandPool();
        void createCommandBuffers();
        void createSampler();
        void createSyncPrimitives();

        void waitForInFlightFence();
        uint32_t acquireNextSwapchainImage();
        void beginCommandBuffer();

        void createMaterialSetLayout();

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

        void submitCommandBuffer(uint32_t imageIndex);
        void presentSwapchainImage(uint32_t imageIndex);
    };
}