#pragma once

#include <glm/glm.hpp>

#include <cstdint>
#include <vector>
#include <string>
#include <filesystem>

namespace engine {
    struct Vertex {
        glm::vec3 position;
        glm::vec3 normal;
        glm::vec2 uv;
    };
    static_assert(sizeof(Vertex) == 32, "Vertex struct must be 32 bytes in size.");

    struct Material {
        glm::vec4 diffuse{0.8f, 0.8f, 0.8f, 1.0f};
        glm::vec4 specular {0.0f};
        glm::vec4 emission {0.0f};
        glm::ivec4 textureIndices {-1, -1, -1, -1};
    };
    static_assert(sizeof(Material) == 64, "Material struct must be 64 bytes in size.");

    struct DrawRange {
        uint32_t firstIndex;
        uint32_t indexCount;
        uint32_t materialIndex;
    };
    // static_assert(sizeof(DrawRange) == 12, "DrawRange struct must be 12 bytes in size.");

    struct TextureRef {
        std::filesystem::path path;
        bool isSRGB = false;
    };

    struct Scene {
        std::string name;
        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;
        std::vector<Material> materials;
        std::vector<std::string> materialNames;
        std::vector<TextureRef> textures;
        std::vector<DrawRange> draws;
        glm::vec3 boundsMin{0.0f};
        glm::vec3 boundsMax{0.0f};
    };

    Scene loadSceneFromFile(const std::filesystem::path& path, float scale = 1.0f);
}