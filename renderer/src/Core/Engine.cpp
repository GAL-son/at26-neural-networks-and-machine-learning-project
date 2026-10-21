#include "Engine.hpp"

#include <iostream>
#include <thread>

Engine::Engine(bool headless): m_headless(headless), m_window(1280, 720, "AI Antialiaser Renderer", headless), m_vulkanContext(m_window) {
    if(!m_headless) {
        initImGui();
    }
}

Engine::~Engine() {
    // Cleanup ImGui / Vulkan
}

void Engine::initImGui() {
    
}

void Engine::run() {
    std::cout << "Starting engine loop in " << (m_headless ? "HEADLESS" : "WINDOWED") << " mode...\n";
    m_lastTime = std::chrono::high_resolution_clock::now();

    while(!m_window.shouldClose()) {
        auto currentTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> delta = currentTime - m_lastTime;
        m_lastTime = currentTime;

        m_frameTimeMs = delta.count();
        m_passTimeMs = m_frameTimeMs + 0.8f; // Mock for now
        m_processedFrames++;

        m_window.poolEvents();

        if (m_headless) {
            // In pure headless training mode, process dataset items in a batch loop
            std::this_thread::sleep_for(std::chrono::milliseconds(16)); // Simulate work
            if (m_processedFrames > 100) {
                std::cout << "Headless batch generation complete. Exiting.\n";
                break;
            }
        } else {
            // Windowed mode with ImGui stats overlay
            // TODO: Render frame & ImGui draw data
        }
    }

    std::cout << "Total processed frames: " << m_processedFrames << "\n";
}