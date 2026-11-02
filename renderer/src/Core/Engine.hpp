#pragma once

#include <chrono>
#include <future>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "Core/Window.hpp"
#include "Scene/Scene.hpp"
#include "Scene/CameraPath.hpp"
#include "Renderer/VulkanContext.hpp"
#include "Renderer/VulkanRenderer.hpp"
#include "IO/ExrFrameWriter.hpp"
#include "UI/UIManager.hpp"

namespace Engine
{
    struct LaunchOptions
    {
        bool headless = false;
        std::filesystem::path configPath = "config.json";
    };

    struct LoadedScene
    {
        std::unique_ptr<Renderer::Scene> gpuScene;
        Scene::CameraPath cameraPath;
    };

    class Engine
    {
    public:
        explicit Engine(const LaunchOptions &options);
        ~Engine();

        int run();

    private:
        static constexpr VkExtent2D kRenderExtent{1920, 1080};

        int runHeadless();
        void runWindowed();
        void renderSceneFrame(ImDrawData *uiDrawData);

        void requestSceneLoad(const std::filesystem::path &objPath);
        void poolSceneLoad();

        void applyCameraPose();
        void advanceAnimation();

        void startRecording();
        void stopRecording();

        void rescanScenes();
        void handleUiActions(const UI::ControlPanelActions &actions);

        LaunchOptions m_options;
        Window m_window;
        VulkanContext m_vulkanContext;
        Renderer::VulkanRenderer m_renderer;
        std::unique_ptr<UI::UIManager> m_ui;

        uint32_t m_processedFrames = 0;
        float m_frameTimeMs = 0.0f;
        float m_passTimeMs = 0.0f;
        std::chrono::high_resolution_clock::time_point m_lastTime;

        std::unique_ptr<Camera> m_camera;

        std::filesystem::path m_scenesDir;
        std::vector<std::filesystem::path> m_sceneFiles;
        std::filesystem::path m_activeScenePath;
        std::string m_sceneError;

        Scene::CameraPath m_cameraPath;
        uint32_t m_animationFrame = 0;

        std::filesystem::path m_outputRoot = "captures";
        std::unique_ptr<IO::ExrFrameWriter> m_frameWriter;
        bool m_recording = false;

        std::future<LoadedScene> m_pendingScene; // Keep last
        std::filesystem::path m_pendingScenePath;
    };
}
