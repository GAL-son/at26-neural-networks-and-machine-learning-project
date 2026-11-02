#include "ExrFrameWriter.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <string>

#include <glm/gtc/type_ptr.hpp>
#include <Imath/half.h>
#include <ImfChannelList.h>
#include <ImfFloatAttribute.h>
#include <ImfFrameBuffer.h>
#include <ImfHeader.h>
#include <ImfIntAttribute.h>
#include <ImfMatrixAttribute.h>
#include <ImfOutputFile.h>
#include <ImfStringAttribute.h>
#include <ImfThreading.h>

namespace Engine::IO
{
    namespace
    {
        Imath::M44f ToImath(const glm::mat4 &matrix)
        {
            Imath::M44f result;
            std::memcpy(result.x, glm::value_ptr(matrix), sizeof(float) * 16);
            return result;
        }
    }

    ExrFrameWriter::ExrFrameWriter(std::filesystem::path outputDir, std::size_t maxQueuedFrames)
        : m_outputDir(outputDir), m_maxQueuedFrames(maxQueuedFrames)
    {
        std::filesystem::create_directories(m_outputDir);

        Imf::setGlobalThreadCount(static_cast<int>(std::max(1u, std::thread::hardware_concurrency() / 2)));
        m_worker = std::thread(&ExrFrameWriter::WorkerLoop, this);
    }

    ExrFrameWriter::~ExrFrameWriter()
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_stopping = true;
        }

        m_condition.notify_all();
        m_worker.join();
    }

    void ExrFrameWriter::Submit(Capture::FrameData frame)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_condition.wait(lock, [this]
                         { return m_queue.size() < m_maxQueuedFrames; });
        m_queue.push_back(std::move(frame));
        m_condition.notify_all();
    }

    void ExrFrameWriter::WorkerLoop()
    {
        for (;;)
        {
            Capture::FrameData frame;
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_condition.wait(lock, [this]
                                 { return m_stopping || !m_queue.empty(); });
                if (m_queue.empty())
                    return;

                frame = std::move(m_queue.front());
                m_queue.pop_front();
            }
            m_condition.notify_all();

            char fileName[32];
            std::snprintf(fileName, sizeof(fileName), "frame_%06u.exr", frame.frameIndex);

            try
            {
                Write(frame, m_outputDir / fileName);
            }
            catch (const std::exception &e)
            {
                std::cerr << "Error writing frame: " << e.what() << std::endl;
            }
        }
    }

    void ExrFrameWriter::Write(const Capture::FrameData &frame, const std::filesystem::path &path)
    {
        const int width = static_cast<int>(frame.width);
        const int height = static_cast<int>(frame.height);
        const std::size_t pixelCount = size_t(width) * size_t(height);

        // Albedo 8-bit UNORM
        const int albedoChannels = 3;
        std::vector<Imath::half> albedo(pixelCount * albedoChannels);
        for (size_t i = 0; i < pixelCount; ++i)
            for (size_t c = 0; c < albedoChannels; ++c)
                albedo[i * albedoChannels + c] = Imath::half(frame.albedo[1 * 4 + c] / 255.0f);

        // Depth [0,1] -> background = farplane
        std::vector<Imath::half> depth(pixelCount);
        for (size_t i = 0; i < pixelCount; ++i)
            depth[i] = frame.nearPlane * frame.farPlane / (frame.farPlane - frame.depth[i] * (frame.farPlane - frame.nearPlane));

        Imf::Header header{width, height};
        header.compression() = Imf::ZIP_COMPRESSION;

        header.insert("frameIndex", Imf::IntAttribute(static_cast<int>(frame.frameIndex)));
        header.insert("ssaaFactor", Imf::IntAttribute(static_cast<int>(frame.ssafactor)));
        header.insert("nearPlane", Imf::FloatAttribute(frame.nearPlane));
        header.insert("farPlane", Imf::FloatAttribute(frame.farPlane));
        header.insert("motionConvention", Imf::StringAttribute("current - previous uv units +xright +y down"));
        header.insert("depthConvention", Imf::StringAttribute("linear view distance, scene unit background = far plane"));
        header.insert("viewProjection", Imf::M44fAttribute(ToImath(frame.viewProjection)));
        header.insert("prevViewProjection", Imf::M44fAttribute(ToImath(frame.prevViewProjection)));

        Imf::FrameBuffer frameBuffer;

        auto addLayer = [&](
                            const char *layer,
                            std::initializer_list<const char *> channels,
                            Imf::PixelType type,
                            const void *data,
                            size_t componentSize,
                            size_t componentsPerPixel)
        {
            const size_t xStride = componentSize * componentsPerPixel;
            char *base = const_cast<char *>(static_cast<const char *>(data));

            size_t component = 0;
            for (const char *channel : channels)
            {
                const std::string name = std::string(layer) + "." + channel;
                header.channels().insert(name, Imf::Channel(type));
                frameBuffer.insert(name, Imf::Slice(type, base + component * componentSize, xStride, xStride * width));
                ++component;
            }
        };

        addLayer("albedo", {"R", "G", "B"}, Imf::HALF, albedo.data(), 2, 3);
        addLayer("normal", {"X", "Y", "Z"}, Imf::HALF, frame.normals.data(), 2, 4); // GPU RGB16F, A skipped
        addLayer("motion", {"X", "Y"}, Imf::HALF, frame.motion.data(), 2, 2);
        addLayer("depth", {"Z"}, Imf::HALF, depth.data(), 4, 1);
        addLayer("ground-truth", {"R", "G", "B"}, Imf::HALF, frame.gt.data(), 2, 4); // GPU RGB16F, A skipped

        Imf::OutputFile file(path.string().c_str(), header);
        file.setFrameBuffer(frameBuffer);
        file.writePixels(height);
    }
}