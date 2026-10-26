#pragma once

#include "Window.hpp"

#include <chrono>

#include "../Scene/Scene.hpp"

#include "../Renderer/VulkanContext.hpp"
#include "../Renderer/VulkanRenderer.hpp"

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
        Renderer::VulkanRenderer m_renderer;
        VulkanContext m_vulkanContext;

        uint32_t m_processedFrames = 0;
        float m_frameTimeMs = 0.0f;
        float m_passTimeMs = 0.0f;
        std::chrono::high_resolution_clock::time_point m_lastTime;
    };
}
