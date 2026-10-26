#include "Scene.hpp"
#include "../Utils/Models/ModelLoader.hpp"

namespace Engine::Scene
{

    Scene::Scene(float aspectWidth, float aspectHeight)
        : m_camera(aspectWidth, aspectHeight)
    {
        // Configure default camera position for the bedroom scene
        m_camera.setPosition(glm::vec3(8.5f, -4.8f, 90.0f));
        m_camera.setRotationEuler(0.0f, 0.0f, 0.0f);
    }

    void Scene::loadModel(const std::string &filepath)
    {
        // Use the model loader utility to construct a CPU-side renderable entity
        RenderableEntity entity = Utils::Models::ModelLoader::loadModel(filepath);
        m_entities.push_back(std::move(entity));
    }

} // namespace Engine::Scene