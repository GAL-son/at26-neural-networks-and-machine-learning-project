#pragma once

#include <string>
#include <glm/glm.hpp>

namespace Engine::Scene
{
    struct Material
    {
        std::string name{"DefaultMaterial"};

        // Phong / Blinn-Phong parameters
        glm::vec3 ambient{0.1f, 0.1f, 0.1f};
        glm::vec3 diffuse{0.8f, 0.8f, 0.8f};
        glm::vec3 specular{0.5f, 0.5f, 0.5f};
        float shininess{32.0f};
        float opacity{1.0f};

        // Texture indices into Scene::textures (-1 means no texture assigned)
        int32_t diffuseTextureIndex{-1};
        int32_t specularTextureIndex{-1};
        int32_t normalTextureIndex{-1};
        int32_t roughnessTextureIndex{-1};

        bool hasDiffuseTexture() const { return diffuseTextureIndex != -1; }
        bool hasSpecularTexture() const { return specularTextureIndex != -1; }
        bool hasNormalTexture() const { return normalTextureIndex != -1; }
        bool hasRoughnessTexture() const { return roughnessTextureIndex != -1; }
    };
}