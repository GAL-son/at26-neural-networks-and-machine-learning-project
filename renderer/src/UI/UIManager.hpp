#pragma once
#define VULKAN_HPP_NO_CONSTRUCTORS

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>
#include <vulkan/vulkan.hpp>

#include "Core/Window.hpp"
#include "Renderer/VulkanContext.hpp"
#include "Utils/Macros.hpp"

namespace Engine::UI
{
    struct ControlPanelState {
        const std::filesystem::path &sceneDir;
        const std::vector<std::filesystem::path> &sceneFiles;
        const std::filesystem::path &activeScene;
        const std::filesystem::path &loadingScene;
        const std::string &error;
        bool generatedPath;
        bool recording;
        uint32_t animationFrame;
        uint32_t animatioFrameCount;
    };

    struct ControlPanelActions {
        bool choseScenesDir = false;
        bool rescan = false;
        std::optional<std::filesystem::path> loadScene;
        bool startRecording = false;
        bool stopRecording = false;
    };

    class UIManager
    {
    public:
        UIManager(const Window &window, const VulkanContext &vulkanContext);
        ~UIManager();

        DISALLOW_COPY(UIManager)
        DISALLOW_MOVE(UIManager)

        void beginFrame();
        void renderStats(uint32_t processedFrames, float frameTimeMs, float passTimeMs);
        void renderBufferLabels();

        ControlPanelActions renderControlPanel(const ControlPanelState &state);
        ImDrawData *endFrame();

        static std::optional<std::filesystem::path> PickDirectory(const std::filesystem::path &start);

    private:
        const VulkanContext &m_context;
        VkDescriptorPool m_descriptorPool{VK_NULL_HANDLE};
        VkFormat m_swapchainFormat;
    };
}