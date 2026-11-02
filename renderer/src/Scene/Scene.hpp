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

        AABB computeBounds() const
        {
            AABB bounds;
            for (const auto &mesh : meshes)
            {
                if(mesh.vertices.empty())
                    continue;

                bounds.grow(mesh.bounds.min);
                bounds.grow(mesh.bounds.max);
            }

            return bounds;
        }

        void clear()
        {
            meshes.clear();
            materials.clear();
            textures.clear();
        }
    };
}