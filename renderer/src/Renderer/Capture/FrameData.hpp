#pragma once

#include <cstdint>
#include <vector>
#include <glm/glm.hpp>

namespace Engine::Capture
{
    struct FrameData
    {
        uint32_t frameIndex = 0;
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t ssafactor = 1;
        float nearPlane = 0.0f;
        float farPlane = 0.0f;

        glm::mat4 viewProjection{1.0f};
        glm::mat4 prevViewProjection{1.0f};

        std::vector<uint8_t> albedo;   // R8G8B8A8_UNORM
        std::vector<uint16_t> normals; // R16G16B16A16_SFLOAT
        std::vector<uint16_t> motion;  // R16G16_SFLOAT
        std::vector<float> depth;      // D32_SFLOAT
        std::vector<uint8_t> gt;       // R16G16B16A16_SFLOAT
    };
} // namesapce Engine::Capture