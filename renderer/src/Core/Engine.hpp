#pragma once

#include "Window.hpp"
#include "../Renderer/VulkanContext.hpp"
#include <chrono>

class Engine
{
public:
    Engine(bool headless);
    ~Engine();

    void run();

private:
    void initImGui();
    void updateStats();

    bool m_headless;
    Window m_window;
    VulkanContext m_vulkanContext;

    uint32_t m_processedFrames = 0;
    float m_frameTimeMs = 0.0f;
    float m_passTimeMs = 0.0f;
    std::chrono::high_resolution_clock::time_point m_lastTime;
};