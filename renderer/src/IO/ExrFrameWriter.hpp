#pragma once
#include "Utils/Macros.hpp"

#include <condition_variable>
#include <deque>
#include <filesystem>
#include <mutex>
#include <thread>

#include "Renderer/Capture/FrameData.hpp"

namespace Engine::IO
{
    class ExrFrameWriter
    {
    public:
        explicit ExrFrameWriter(std::filesystem::path outputDir, std::size_t maxQueuedFrames = 4);
        ~ExrFrameWriter();

        DISALLOW_COPY(ExrFrameWriter)

        void Submit(Capture::FrameData frame);
        static void Write(const Capture::FrameData &frame, const std::filesystem::path &path);

    private:
        void WorkerLoop();

        std::filesystem::path m_outputDir;
        std::size_t m_maxQueuedFrames;

        std::deque<Capture::FrameData> m_queue;
        std::mutex m_mutex;
        std::condition_variable m_condition;
        bool m_stopping = false;

        std::thread m_worker;
    };
}