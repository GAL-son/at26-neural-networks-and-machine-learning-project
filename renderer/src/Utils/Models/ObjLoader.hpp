#pragma once

#include <string>
#include "Scene/Scene.hpp"

namespace Engine::Utils
{
    class ObjLoader
    {
    public:

        static Scene::Scene loadModel(const std::string& filepath);
    };

} // namespace Engine::Scene