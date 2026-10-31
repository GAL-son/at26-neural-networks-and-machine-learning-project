#include "ObjLoader.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <stdexcept>
#include <iostream>
#include <unordered_map>
#include <filesystem>

#include "Utils/Textures/TextureLoader.hpp"

namespace Engine::Utils
{
    Scene::Scene ObjLoader::loadModel(const std::string &filepath)
    {
        Scene::Scene scene;
        scene.name = std::filesystem::path(filepath).stem().string();

        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;

        // Extract base directory for material and texture loading
        std::string baseDir = filepath.substr(0, filepath.find_last_of("/\\") + 1);

        bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filepath.c_str(), baseDir.c_str());

        if (!warn.empty())
        {
            std::cout << "TinyObj Warning: " << warn << "\n";
        }
        if (!err.empty())
        {
            std::cerr << "TinyObj Error: " << err << "\n";
        }
        if (!ret)
        {
            throw std::runtime_error("Failed to load OBJ file: " + filepath);
        }

        // --- 1. Parse Materials ---
        scene.materials.reserve(materials.size());
        std::unordered_map<std::string, int32_t> loadedTextureCache;

        auto loadTextureIfNeeded = [&](const std::string &texname) -> int32_t
        {
            if (texname.empty())
            {
                return -1;
            }

            std::string fullPath = baseDir + texname;

            // Check if we already loaded this texture during this import
            if (loadedTextureCache.find(fullPath) != loadedTextureCache.end())
            {
                return loadedTextureCache[fullPath];
            }

            // Load texture from disk using your TextureLoader
            Scene::Texture texture = TextureLoader::loadFromFile(fullPath);

            // Push into the scene's central texture library
            scene.textures.push_back(std::move(texture));

            int32_t textureIndex = static_cast<int32_t>(scene.textures.size() - 1);
            loadedTextureCache[fullPath] = textureIndex;

            return textureIndex;
        };

        for (const auto &mat : materials)
        {
            Scene::Material material;
            material.name = mat.name;

            // Colors
            material.ambient = glm::vec3(mat.ambient[0], mat.ambient[1], mat.ambient[2]);
            material.diffuse = glm::vec3(mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]);
            material.specular = glm::vec3(mat.specular[0], mat.specular[1], mat.specular[2]);
            material.shininess = mat.shininess;
            material.opacity = mat.dissolve; // d or Tr in .mtl

            // Textures (Loads CPU pixels and assigns index pointers)
            material.diffuseTextureIndex = loadTextureIfNeeded(mat.diffuse_texname);
            material.specularTextureIndex = loadTextureIfNeeded(mat.specular_texname);
            material.normalTextureIndex = loadTextureIfNeeded(mat.bump_texname);
            // material.roughnessTextureIndex = loadTextureIfNeeded(mat.roughness_texname); // if supported by tinyobj

            scene.materials.push_back(material);
        }

        // Fallback material if none were defined in the .mtl
        if (scene.materials.empty())
        {
            scene.materials.push_back(Scene::Material{"DefaultMaterial"});
        }

        // --- 2. Parse Geometry & Group by Material into Submeshes ---
        Scene::Mesh mesh;
        mesh.name = scene.name + "_Mesh";

        // Temporary structure to hold index batches per material ID
        std::unordered_map<int32_t, std::vector<Scene::Index>> materialBatches;

        for (const auto &shape : shapes)
        {
            size_t index_offset = 0;
            for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++)
            {
                int fv = shape.mesh.num_face_vertices[f];
                int32_t matId = shape.mesh.material_ids[f];

                // Ensure material index is valid, otherwise fallback to 0
                if (matId < 0 || matId >= static_cast<int32_t>(scene.materials.size()))
                {
                    matId = 0;
                }

                // Process each vertex in the face (assumes triangulation or handles general polygons)
                for (int v = 0; v < fv; v++)
                {
                    tinyobj::index_t idx = shape.mesh.indices[index_offset + v];

                    Scene::Vertex vertex{};

                    // Position (mandatory)
                    vertex.position = {
                        attrib.vertices[3 * idx.vertex_index + 0],
                        attrib.vertices[3 * idx.vertex_index + 1],
                        attrib.vertices[3 * idx.vertex_index + 2]};

                    // Normal (optional)
                    if (idx.normal_index >= 0)
                    {
                        vertex.normal = {
                            attrib.normals[3 * idx.normal_index + 0],
                            attrib.normals[3 * idx.normal_index + 1],
                            attrib.normals[3 * idx.normal_index + 2]};
                    }
                    else
                    {
                        vertex.normal = {0.0f, 1.0f, 0.0f}; // Fallback normal
                    }

                    // Texture Coordinates (optional - note V coordinate flip for Vulkan)
                    if (idx.texcoord_index >= 0)
                    {
                        vertex.texCoord = {
                            attrib.texcoords[2 * idx.texcoord_index + 0],
                            1.0f - attrib.texcoords[2 * idx.texcoord_index + 1]};
                    }

                    // Add vertex and record its index in the global vertex buffer
                    mesh.vertices.push_back(vertex);
                    Scene::Index newVertexIndex = static_cast<Scene::Index>(mesh.vertices.size() - 1);

                    materialBatches[matId].push_back(newVertexIndex);
                }
                index_offset += fv;
            }
        }

        // --- 3. Build Submeshes from Material Batches ---
        for (const auto &[matId, batchIndices] : materialBatches)
        {
            Scene::Submesh submesh;
            submesh.name = "Submesh_Mat_" + std::to_string(matId);
            submesh.indexOffset = static_cast<Scene::Index>(mesh.indices.size());
            submesh.indexCount = static_cast<Scene::Index>(batchIndices.size());
            submesh.materialIndex = matId;

            // Append batch indices to the master index buffer
            mesh.indices.insert(mesh.indices.end(), batchIndices.begin(), batchIndices.end());
            mesh.submeshes.push_back(submesh);
        }

        // Compute local bounding box (AABB) for culling/camera placement
        mesh.computeBounds();

        // Add the single loaded mesh to the scene
        scene.meshes.push_back(mesh);

        std::cout << "[ObjectLoader] Successfully loaded: " << filepath << "\n"
                  << "  - Vertices: " << mesh.vertices.size() << "\n"
                  << "  - Indices:  " << mesh.indices.size() << "\n"
                  << "  - Submeshes:" << mesh.submeshes.size() << "\n"
                  << "  - Materials:" << scene.materials.size() << "\n";

        return scene;
    }

} // namespace Engine::Utils