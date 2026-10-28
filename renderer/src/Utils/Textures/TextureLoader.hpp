#pragma once

#include <string>
#include "Scene/Texture.hpp"

namespace Engine::Utils
{
    class TextureLoader
    {
    public:
        static Scene::Texture loadFromFile(const std::string& filepath);
    };
} // namespace Engine::Scene