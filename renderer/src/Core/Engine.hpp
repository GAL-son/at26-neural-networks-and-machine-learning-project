#pragma once

#include "Window.hpp"
#include "../Renderer/VulkanContext.hpp"
#include "../UI/UIManager.hpp"
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
    void initOffscreenBuffers();
    void cleanupOffscreenBuffers();
    void initCompositePipeline();
    VkShaderModule Engine::createShaderModule(const std::vector<char>& code);
    

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

    struct OffscreenBuffer
    {
        VkImage image = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        VkImageView imageView = VK_NULL_HANDLE;
        VkSampler sampler = VK_NULL_HANDLE;
    };
    OffscreenBuffer m_buffers[4];

    VkDescriptorSetLayout m_compositeDescriptorLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_compositeDescriptorSet = VK_NULL_HANDLE;
    VkPipelineLayout m_compositePipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_compositePipeline = VK_NULL_HANDLE;
    VkDescriptorPool m_compositePool = VK_NULL_HANDLE;
};