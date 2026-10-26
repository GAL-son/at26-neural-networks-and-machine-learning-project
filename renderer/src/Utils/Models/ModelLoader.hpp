#pragma once

#include <string>
#include <memory>
#include "../../Scene/Scene.hpp"

namespace Utils::Models
{
    struct LoadedMaterialData
    {
        std::string name;
        std::string diffuseTexName;
        glm::vec3 diffuseColor = glm::vec3(1.0f);
    };

    struct LoadedSubmeshData
    {
        uint32_t indexCount = 0;
        uint32_t firstIndex = 0;
        int materialIndex = -1;
    };

    struct LoadedMeshData
    {
        std::vector<float> positions;
        std::vector<float> normals;
        std::vector<float> texCoords;
        std::vector<uint32_t> indices;
        std::vector<LoadedSubmeshData> submeshes;
        std::vector<LoadedMaterialData> materials;
        std::string baseDir;
    };

    // Goła funkcja zamiast klasy z metodą statyczną
    Engine::Scene::RenderableEntity loadModel(const std::string &filepath);

} // namespace Utils::Models