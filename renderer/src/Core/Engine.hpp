#pragma once

#include "Window.hpp"

#include <chrono>

#include "../Scene/Scene.hpp"

#include "Renderer/VulkanContext.hpp"
#include "Renderer/VulkanRenderer.hpp"
#include "IO/ExrFrameWriter.hpp"

namespace Engine
{
    class Engine
    {
    public:
        Engine(bool headless);
        ~Engine();

        void run();

    private:
        void runHeadless();
        void runWindowed();

        std::unique_ptr<Scene::Scene> m_scene;

        bool m_headless;
        Window m_window;
        VulkanContext m_vulkanContext;
        Renderer::VulkanRenderer m_renderer;

        uint32_t m_processedFrames = 0;
        float m_frameTimeMs = 0.0f;
        float m_passTimeMs = 0.0f;
        std::chrono::high_resolution_clock::time_point m_lastTime;

        std::unique_ptr<Camera> m_camera;

        std::unique_ptr<IO::ExrFrameWriter> m_frameWriter;
        bool m_recording = false;
    };
}
