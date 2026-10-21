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
    void renderImGuiOverlay();
    void updateStats();

    bool m_headless;
    Window m_window;
    VulkanContext m_vulkanContext;

    uint32_t m_processedFrames = 0;
    float m_frameTimeMs = 0.0f;
    float m_passTimeMs = 0.0f;
    std::chrono::high_resolution_clock::time_point m_lastTime;

    // Command pool for UI / presentation
    VkCommandPool m_commandPool = VK_NULL_HANDLE;
    VkDescriptorPool m_imguiPool = VK_NULL_HANDLE;
    
};