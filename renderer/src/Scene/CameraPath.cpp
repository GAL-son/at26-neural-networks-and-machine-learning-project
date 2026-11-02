#include "CameraPath.hpp"

#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <glm/gtc/constants.hpp>
#include <nlohmann/json.hpp>

namespace Engine::Scene
{
    CameraPath CameraPath::ForScene(const std::filesystem::path &objPath, const AABB &bounds)
    {
        std::cout << "[Camera path] scene bounds min ("
                  << bounds.min.x << ", " << bounds.min.y << ", " << bounds.min.z
                  << ") max (" << bounds.max.x << ", " << bounds.max.y << ", " << bounds.max.z << ")" << std::endl;

        const std::filesystem::path sidecar = objPath.parent_path() / "animation.json";
        if (std::filesystem::exists(sidecar))
        {
            CameraPath path = LoadJson(sidecar);
            std::cout << "[CameraPath] laoded " << path.frames.size() << "frames from " << sidecar << std::endl;
            return path;
        }

        CameraPath path = GenerateOrbit(bounds);
        std::cout << "[CameraPath] no animation.json! Generated orbit with " << path.frames.size() << "frames" << std::endl;
        return path;
    }

    CameraPath CameraPath::LoadJson(const std::filesystem::path &file)
    {
        std::ifstream stream(file);
        if (!stream)
            throw std::runtime_error("[CameraPath] failed to open file" + file.string());

        CameraPath path;
        try
        {
            const nlohmann::json json = nlohmann::json::parse(stream);
            for (const auto &frame : json.at("frames"))
            {
                const auto p = frame.at("position").get<std::array<float, 3>>();
                ;
                CameraPose pose;
                pose.position = {p[0], p[1], p[2]};
                pose.pitch = glm::clamp(frame.at("pitch").get<float>(), -89.0f, 89.0f);
                pose.yaw = frame.at("yaw").get<float>();
                path.frames.push_back(pose);
            }
        }
        catch (const nlohmann::json::exception &e)
        {
            throw std::runtime_error(file.string() + ": " + e.what());
        }

        if (path.frames.empty())
        {
            throw std::runtime_error(file.string() + ": 'frames' is empty!");
        }
        return path;
    }

    CameraPath CameraPath::GenerateOrbit(const AABB &bounds, uint32_t frameCount, float margin)
    {
        constexpr float kInwardLookDeg = 30.0f;
        constexpr float kPitchDeg = -10.0f;
        constexpr float kPitchSwingDeg = 5.0f;
        constexpr float kHeightSwing = 0.15f;

        const glm::vec3 size = bounds.max - bounds.min;

        if (!(size.x > 0.0f && size.z > 0.0f))
            throw std::runtime_error("Invalid bounds. Scene has no extent");

        const glm::vec3 innerMin = bounds.min + size * margin;
        const glm::vec3 innerMax = bounds.max - size * margin;
        const glm::vec3 center = 0.5f * (innerMin + innerMax);
        const glm::vec3 radius = 0.5f * (innerMin - innerMax);

        CameraPath path;
        path.generated = true;
        path.frames.reserve(frameCount);

        for (uint32_t i = 0; i < frameCount; ++i)
        {
            const float theta = glm::two_pi<float>() * float(i) / float(frameCount);

            CameraPose pose;
            pose.position = center + glm::vec3(
                                         radius.x * std::cos(theta),
                                         radius.y * kHeightSwing * std::sin(2.0f * theta),
                                         radius.z * std::sin(theta));

            const float dx = -radius.x * std::sin(theta);
            const float dz = radius.z * std::cos(theta);
            pose.yaw = glm::degrees(std::atan2(dz, dx)) + kInwardLookDeg;
            pose.pitch = kPitchDeg + kPitchSwingDeg + std::sin(theta);

            path.frames.push_back(pose);            
        }

        return path;
    }
} // namespace Engine::Scene
