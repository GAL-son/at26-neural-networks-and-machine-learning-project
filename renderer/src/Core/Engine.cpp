#include "Engine.hpp"
#include <iostream>
#include <thread>

namespace Engine
{

    Engine::Engine(bool headless)
        : m_headless(headless),
          m_window(1280, 720, "AI Antialiaser Renderer", headless),
          m_vulkanContext(m_window),
          m_renderer(m_vulkanContext, m_window)
    {
        if (!m_headless)
        {
            m_renderer.init();

            // --- Initialize Scene & Model ---
            VkExtent2D extent = m_vulkanContext.getSwapchainExtent();
            m_scene = std::make_unique<Scene::Scene>(
                m_vulkanContext.getAllocator(),
                static_cast<float>(extent.width),
                static_cast<float>(extent.height));

            try
            {
                m_scene->loadModel("scenes/bedroom/iscv2.obj");
                std::cout << "Successfully loaded and converted McGuire bedroom scene!\n";
            }
            catch (const std::exception &e)
            {
                std::cerr << "Failed to load model: " << e.what() << std::endl;
            }
        }
    }

    Engine::~Engine() = default; // Destructor is clean since m_renderer and m_scene handle their own RAII cleanup!

    void Engine::run()
    {
        std::cout << "Running loop in " << (m_headless ? "HEADLESS" : "WINDOWED") << " mode...\n";
        m_lastTime = std::chrono::high_resolution_clock::now();

        if (m_headless)
        {
            runHeadless();
        }
        else
        {
            runWindowed();
        }
    }

    void Engine::runHeadless()
    {
        while (!m_window.shouldClose())
        {
            auto currentTime = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float, std::milli> delta = currentTime - m_lastTime;
            m_lastTime = currentTime;

            m_frameTimeMs = delta.count();
            m_passTimeMs = m_frameTimeMs * 0.75f;
            m_processedFrames++;

            m_window.pollEvents();
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
            if (m_processedFrames > 150)
                break;
        }
        std::cout << "Headless batch generation complete. Exiting.\n";
    }

    void Engine::runWindowed()
    {
        while (!m_window.shouldClose())
        {
            auto currentTime = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float, std::milli> delta = currentTime - m_lastTime;
            m_lastTime = currentTime;

            m_frameTimeMs = delta.count();
            m_passTimeMs = m_frameTimeMs * 0.75f;
            m_processedFrames++;

            m_window.pollEvents();

            // Delegate entire frame rendering to the renderer domain
            m_renderer.renderFrame(m_scene.get(), m_processedFrames, m_frameTimeMs, m_passTimeMs);
        }
    }
}
