#include "Engine.hpp"
#include <iostream>
#include <thread>

#include "Utils/Models/ObjLoader.hpp"

namespace Engine
{
    Engine::Engine(bool headless)
        : m_headless(headless),
          m_window(1280, 720, "AI Antialiaser Renderer", headless),
          m_vulkanContext(m_window),
          m_renderer(m_vulkanContext, m_window)
    {
        m_renderer.init();

        // --- Initialize Scene & Model on CPU ---
        VkExtent2D extent = m_vulkanContext.getSwapchainExtent();
        float aspectRatio = static_cast<float>(extent.width) / static_cast<float>(extent.height);

        // FOV = 45 stopni, near = 0.1f, far = 100.0f
        m_camera = std::make_unique<Camera>(80.0f, aspectRatio, 0.1f, 10000.0f);

        // Ustawiamy kamerę tak, żeby patrzyła na scenę (np. lekko cofnięta i w górę)
        m_camera->setPosition(glm::vec3(0.0f, 0.0f, 0.0f));

        try
        {
            // 1. Wczytujemy scenę z pliku za pomocą statycznej metody ObjLoader
            // Scene::Scene cpuScene = Utils::ObjLoader::loadModel("scenes/sponza/sponza.obj");
            Scene::Scene cpuScene = Utils::ObjLoader::loadModel("scenes/bedroom/iscv2.obj");

            // 2. Tworzymy unikalny wskaźnik na naszą scenę CPU
            m_scene = std::make_unique<Scene::Scene>(std::move(cpuScene));
            std::cout << "Successfully loaded McGuire bedroom scene via ObjLoader!\n";

            // 3. Przekazujemy wczytaną scenę do renderera w celu konwersji i uploadu na GPU
            m_renderer.loadScene(*m_scene);
            std::cout << "Successfully converted and uploaded scene assets to GPU!\n";
        }
        catch (const std::exception &e)
        {
            std::cerr << "Failed to load or upload model: " << e.what() << std::endl;
        }
    }

    Engine::~Engine() = default; // RAII dba o czyszczenie zasobów w m_renderer i m_scene!

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
        float cameraRotationDeg = 0;
        m_camera->setPosition(glm::vec3(10.0f, 10.0f, 10.0f));

        while (!m_window.shouldClose())
        {
            m_camera->updatePreviousVp(); // this saves current vp
            m_camera->setRotation(0, cameraRotationDeg); // this updates rotation so also should modify vp right?

            auto currentTime = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float, std::milli> delta = currentTime - m_lastTime;
            m_lastTime = currentTime;

            m_frameTimeMs = delta.count();
            m_passTimeMs = m_frameTimeMs;
            m_processedFrames++;

            m_window.pollEvents();

            // Aktualizacja logiki kamery (np. ruch, rotacja - jeśli zaimplementowana w klasie Engine)
            // m_camera.update(m_frameTimeMs);

            // Delegujemy renderowanie klatki do renderera, przekazując kamerę oraz statystyki UI
            m_renderer.renderFrame(*m_camera, m_processedFrames, m_frameTimeMs, m_passTimeMs);

            
            cameraRotationDeg += 5 * m_passTimeMs * 0.001;
            if (cameraRotationDeg > 360)
                cameraRotationDeg = 0;
        }
    }
}