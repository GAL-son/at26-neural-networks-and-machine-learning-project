#pragma once

#include "Mesh.hpp"
#include "Camera.hpp"
#include <vector>
#include <string>

namespace Engine::Scene
{

    struct RenderableEntity
    {
        Mesh meshData;
        std::vector<Material> materials;
        glm::mat4 transform{1.0f};
    };

    class Scene
    {
    public:
        Scene(float aspectWidth, float aspectHeight);
        ~Scene() = default;

        void loadModel(const std::string &filepath);

        const std::vector<RenderableEntity> &getEntities() const { return m_entities; }
        Camera &getCamera() { return m_camera; }
        const Camera &getCamera() const { return m_camera; }

    private:
        Camera m_camera;
        std::vector<RenderableEntity> m_entities;
    };

} // namespace Engine::Scene