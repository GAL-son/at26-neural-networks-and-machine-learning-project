#pragma once

#include <string>
#include <vector>
#include "Mesh.hpp"
#include "Material.hpp"
#include "Texture.hpp"

namespace Engine::Scene
{
    class Scene
    {
    public:
        std::string name{"DefaultScene"};

        std::vector<Mesh> meshes;
        std::vector<Material> materials;
        std::vector<Texture> textures;
        
        void clear()
        {
            meshes.clear();
            materials.clear();
            textures.clear();
        }
    };
}