#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>

namespace Engine::UI
{
    class ConsoleProgress
    {
    public:
        explicit ConsoleProgress(size_t sceneCount) : m_sceneCount(sceneCount) {};

        void beginScene(size_t index, std::string name)
        {
            m_index = index;
            m_name = std::move(name);
            m_start = Clock::now();
            m_lastPrint = {};
        }

        void loading()
        {
            if (!throtle())
                return;

            static const char kSpinner[] = "|/-\\|";
            const float t = elapsed();
            std::printf("\r[%zu/%zu] %s loading %c %.0fs       ",
                        m_index + 1, m_sceneCount, m_name.c_str(), kSpinner[int(t * 8.0f) % 4], t);
        }

        void recordingStarted()
        {
            std::printf("\n");
            m_start = Clock::now();
            m_lastPrint = {};
        }

        void recording(uint32_t done, uint32_t total)
        {
            if (!throtle() && done != total)
                return;

            constexpr int kBarWidth = 30;
            const float t = elapsed();
            const float fps = t > 0.0f ? done / t : 0.0f;
            const int eta = fps > 0.0f ? int((total - done) / fps) : 0;
            const int filled = total ? int(uint64_t(kBarWidth) * done / total) : 0;

            std::printf("\r[%zu/%zu] %s recording [%s%s] %u/%us ETA: %dm %ds %.0f FPS",
                        m_index + 1,
                        m_sceneCount,
                        m_name.c_str(),
                        std::string(filled, '#').c_str(),
                        std::string(kBarWidth - filled, '-').c_str(),
                        done,
                        total,
                        eta / 60,
                        eta % 60,
                        fps);
            std::fflush(stdout);
        }

        void finished(uint32_t total)
        {
            std::printf("\r[%zu/%zu] %s done: %u frames in %.0fs%40s\n", m_index + 1, m_sceneCount, m_name.c_str(), total, elapsed(), "");
        }

        void failed(const std::string &error)
        {
            std::printf("\r[%zu/%zu] %s failed: %s\n", m_index + 1, m_sceneCount, m_name.c_str(), error.c_str());
        }

    private:
        using Clock = std::chrono::steady_clock;

        bool throtle()
        {
            const auto now = Clock::now();
            if (now - m_lastPrint < std::chrono::milliseconds(200))
                return false;

            m_lastPrint = now;
            return true;
        }

        float elapsed() const { return std::chrono::duration<float>(Clock::now() - m_start).count(); }

        size_t m_sceneCount;
        size_t m_index = 0;
        std::string m_name;
        Clock::time_point m_start{};
        Clock::time_point m_lastPrint{};
    };
}
