#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <limits>
#include <glm/glm.hpp>

namespace Engine::Scene
{
    struct Vertex
    {
        glm::vec3 position{0.0f};
        glm::vec3 normal{0.0f, 1.0f, 0.0f};
        glm::vec2 texCoord{0.0f};
        glm::vec3 tangent{0.0f};
    };

    using Index = uint32_t;

    struct Submesh
    {
        std::string name;
        Index indexOffset{0};
        Index indexCount{0};
        int32_t materialIndex{-1};
    };

    struct AABB
    {
        glm::vec3 min{std::numeric_limits<float>::max()};
        glm::vec3 max{std::numeric_limits<float>::lowest()};

        void grow(const glm::vec3 &point)
        {
            min = glm::min(min, point);
            max = glm::max(max, point);
        }
    };

    class Mesh
    {
    public:
        std::string name;
        std::vector<Vertex> vertices;
        std::vector<Index> indices;
        std::vector<Submesh> submeshes;
        AABB bounds;

        void computeBounds()
        {
            bounds = AABB{};
            for (const auto &vertex : vertices)
            {
                bounds.grow(vertex.position);
            }
        }
    };
}