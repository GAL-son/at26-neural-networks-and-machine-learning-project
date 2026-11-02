#pragma once

#include <vector>
#include <filesystem>
#include <cstdint>
#include <glm/glm.hpp>

#include "Scene/Mesh.hpp"

namespace Engine::Scene
{
    struct CameraPose
    {
        glm::vec3 position{0.0f};
        float pitch = 0.0f;
        float yaw = -90.0f;
    };

    struct CameraPath
    {
        std::vector<CameraPose> frames;
        bool generated = false;

        bool empty() const { return frames.empty(); };

        static CameraPath ForScene(const std::filesystem::path &objPath, const AABB &bounds);
        static CameraPath LoadJson(const std::filesystem::path &file);

        static CameraPath GenerateOrbit(const AABB &bounds, uint32_t frameCount = 600, float margin = 0.15f);
    };
}