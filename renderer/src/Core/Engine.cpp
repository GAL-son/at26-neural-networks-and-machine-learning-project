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
          m_renderer(m_vulkanContext, m_window, 1920, 1080)
    {
        m_renderer.init();

        VkExtent2D extent = m_vulkanContext.getSwapchainExtent();
        float aspectRatio = static_cast<float>(extent.width) / static_cast<float>(extent.height);

        m_camera = std::make_unique<Camera>(80.0f, aspectRatio, 0.1f, 10000.0f);
        m_camera->setPosition(glm::vec3(0.0f, 0.0f, 0.0f));

        requestSceneLoad("scenes/bedroom/iscv2.obj");
        // requestSceneLoad("scenes/sponza/sponza.obj");
    }

    Engine::~Engine() = default; // RAII dba o czyszczenie zasobów w m_renderer i m_scene!

    void Engine::requestSceneLoad(const std::filesystem::path &objPath)
    {
        if (m_pendingScene.valid())
            return;

        m_renderer.SetScene(nullptr);
        m_cameraPath = {};
        m_animationFrame = 0;
        m_pendingScenePath = objPath;

        m_pendingScene = std::async(std::launch::async, [this, objPath]
                                    {
                                    const Scene::Scene cpuScene = Utils::ObjLoader::loadModel(objPath.string());

                                    LoadedScene loaded;
                                    loaded.cameraPath = Scene::CameraPath::ForScene(objPath, cpuScene.computeBounds());
                                    loaded.gpuScene = m_renderer.createGpuScene(cpuScene);
                                    return loaded; });
    }

    void Engine::poolSceneLoad()
    {
        if (!m_pendingScene.valid() || m_pendingScene.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
            return;

        try
        {
            LoadedScene loaded = m_pendingScene.get();
            m_renderer.SetScene(std::move(loaded.gpuScene));
            m_cameraPath = std::move(loaded.cameraPath);
            m_animationFrame = 0;
            std::cout << "Scene ready: " << m_pendingScenePath << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cout << "Exception thrown while loading scene: " << e.what() << std::endl;
        }
        m_pendingScenePath.clear();
    }

    void Engine::applyCameraPose()
    {
        if (m_cameraPath.empty())
            return;

        const Scene::CameraPose &pose = m_cameraPath.frames[m_animationFrame];
        m_camera->updatePreviousVp();
        m_camera->setPosition(pose.position);
        m_camera->setRotation(pose.pitch, pose.yaw);

        if (m_animationFrame == 0)
        {
            m_camera->updatePreviousVp();
        }
    }

    void Engine::advanceAnimation()
    {
        if (m_cameraPath.empty())
            return;

        if (++m_animationFrame == m_cameraPath.frames.size())
        {
            m_animationFrame = 0;

            if(m_recording) {
                m_recording = false;
                m_frameWriter.reset();
                std::cout << "Finished Recording " << m_cameraPath.frames.size() << " frames" << std::endl;
            }
        }
    }

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
        float cameraPositionX = 0;
        m_camera->setPosition(glm::vec3(cameraPositionX, 20.0f, 20.0f));

        while (!m_window.shouldClose())
        {
            auto currentTime = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float, std::milli> delta = currentTime - m_lastTime;
            m_lastTime = currentTime;

            m_frameTimeMs = delta.count();
            m_passTimeMs = m_frameTimeMs;
            m_processedFrames++;

            m_window.pollEvents();
            poolSceneLoad();

            applyCameraPose();
            auto frame = m_renderer.renderFrame(*m_camera, m_processedFrames, m_frameTimeMs, m_passTimeMs, m_recording);
            if (frame)
            {
                if (!m_frameWriter)
                    m_frameWriter = std::make_unique<IO::ExrFrameWriter>("captures");
                frame->frameIndex = m_animationFrame;
                m_frameWriter->Submit(std::move(*frame));
            }

            advanceAnimation();
        }
    }
}