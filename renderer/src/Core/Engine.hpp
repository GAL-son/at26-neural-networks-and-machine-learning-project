#pragma once

#include "Window.hpp"

#include <chrono>

#include "../Renderer/VulkanContext.hpp"
#include "../Renderer/CompositePass.hpp"
#include "../Renderer/GeometryPass.hpp"
#include "../UI/UIManager.hpp"
#include "../Scene/Scene.hpp"

class Engine
{
public:
    Engine(bool headless);
    ~Engine();

    void run();

private:
    void runHeadless();
    void runWindowed();

    std::unique_ptr<Scene> m_scene;

    bool m_headless;
    Window m_window;
    VulkanContext m_vulkanContext;
    UIManager m_uiManager;

    uint32_t m_processedFrames = 0;
    float m_frameTimeMs = 0.0f;
    float m_passTimeMs = 0.0f;
    std::chrono::high_resolution_clock::time_point m_lastTime;

    // Command pool for UI / presentation
    VkCommandPool m_commandPool = VK_NULL_HANDLE;
    VkDescriptorPool m_imguiPool = VK_NULL_HANDLE;

    // The composite pass module handles all offscreen buffers & background rendering
    CompositePass m_compositePass;
    GeometryPass m_geometryPass;
};