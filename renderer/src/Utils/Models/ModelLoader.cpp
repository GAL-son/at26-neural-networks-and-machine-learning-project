#include "ModelLoader.hpp"
#include "FileUtils.hpp" // Contains the existing Utils::loadOBJ and LoadedMeshData structures
#include <stdexcept>
#include <iostream>

namespace Utils::Models
{
    inline bool loadOBJ(const std::string &filepath, LoadedMeshData &outData)
    {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;

        // Extract base directory for material/texture loading
        outData.baseDir = filepath.substr(0, filepath.find_last_of("/\\") + 1);

        bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filepath.c_str(), outData.baseDir.c_str());

        if (!warn.empty())
        {
            std::cout << "TinyObj Warning: " << warn << std::endl;
        }
        if (!err.empty())
        {
            std::cerr << "TinyObj Error: " << err << std::endl;
        }
        if (!ret)
        {
            std::cerr << "Failed to load OBJ file: " << filepath << std::endl;
            return false;
        }

        // 1. Convert tinyobj materials to our LoadedMaterialData format
        for (const auto &mat : materials)
        {
            LoadedMaterialData myMat{};
            myMat.name = mat.name;
            myMat.diffuseTexName = mat.diffuse_texname;
            myMat.diffuseColor = glm::vec3(mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]);
            outData.materials.push_back(myMat);
        }

        // If no materials were defined in the .mtl, provide a default fallback material
        if (outData.materials.empty())
        {
            outData.materials.push_back({"default", "", glm::vec3(1.0f)});
        }

        // 2. Process shapes and group triangles by material into submeshes
        // To handle material grouping cleanly while flattening vertices, we group faces by material ID.
        // We'll organize indices per material first, then append them to outData.indices.

        struct MaterialBatch
        {
            std::vector<uint32_t> batchIndices;
        };

        // Map from material index -> batch of indices
        std::unordered_map<int, std::vector<uint32_t>> materialBatches;

        for (const auto &shape : shapes)
        {
            size_t index_offset = 0;
            for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++)
            {
                int fv = shape.mesh.num_face_vertices[f];
                int matId = shape.mesh.material_ids[f];
                if (matId < 0 || matId >= static_cast<int>(outData.materials.size()))
                {
                    matId = 0; // Fallback to first material
                }

                // Collect vertices for this face
                tinyobj::index_t idx0 = shape.mesh.indices[index_offset + 0];
                tinyobj::index_t idx1 = shape.mesh.indices[index_offset + 1];
                tinyobj::index_t idx2 = shape.mesh.indices[index_offset + 2];

                glm::vec3 v0(attrib.vertices[3 * idx0.vertex_index + 0], attrib.vertices[3 * idx0.vertex_index + 1], attrib.vertices[3 * idx0.vertex_index + 2]);
                glm::vec3 v1(attrib.vertices[3 * idx1.vertex_index + 0], attrib.vertices[3 * idx1.vertex_index + 1], attrib.vertices[3 * idx1.vertex_index + 2]);
                glm::vec3 v2(attrib.vertices[3 * idx2.vertex_index + 0], attrib.vertices[3 * idx2.vertex_index + 1], attrib.vertices[3 * idx2.vertex_index + 2]);

                glm::vec3 faceNormal = glm::normalize(glm::cross(v1 - v0, v2 - v0));

                for (int v = 0; v < fv; v++)
                {
                    tinyobj::index_t index = shape.mesh.indices[index_offset + v];

                    // Position
                    outData.positions.push_back(attrib.vertices[3 * index.vertex_index + 0]);
                    outData.positions.push_back(attrib.vertices[3 * index.vertex_index + 1]);
                    outData.positions.push_back(attrib.vertices[3 * index.vertex_index + 2]);

                    // Normal
                    if (index.normal_index >= 0)
                    {
                        outData.normals.push_back(attrib.normals[3 * index.normal_index + 0]);
                        outData.normals.push_back(attrib.normals[3 * index.normal_index + 1]);
                        outData.normals.push_back(attrib.normals[3 * index.normal_index + 2]);
                    }
                    else
                    {
                        outData.normals.push_back(faceNormal.x);
                        outData.normals.push_back(faceNormal.y);
                        outData.normals.push_back(faceNormal.z);
                    }

                    // TexCoord (Flip V for Vulkan)
                    if (index.texcoord_index >= 0)
                    {
                        outData.texCoords.push_back(attrib.texcoords[2 * index.texcoord_index + 0]);
                        outData.texCoords.push_back(1.0f - attrib.texcoords[2 * index.texcoord_index + 1]);
                    }
                    else
                    {
                        outData.texCoords.push_back(0.0f);
                        outData.texCoords.push_back(0.0f);
                    }

                    uint32_t newVertexIndex = static_cast<uint32_t>(outData.positions.size() / 3 - 1);
                    materialBatches[matId].push_back(newVertexIndex);
                }
                index_offset += fv;
            }
        }

        // 3. Flatten batches into the final global index buffer and build Submesh descriptors
        for (const auto &[matId, batchIndices] : materialBatches)
        {
            LoadedSubmeshData submesh{};
            submesh.firstIndex = static_cast<uint32_t>(outData.indices.size());
            submesh.indexCount = static_cast<uint32_t>(batchIndices.size());
            submesh.materialIndex = matId;

            outData.indices.insert(outData.indices.end(), batchIndices.begin(), batchIndices.end());
            outData.submeshes.push_back(submesh);
        }

        std::cout << "Successfully loaded OBJ & MTL: " << filepath
                  << " (" << outData.indices.size() << " indices, "
                  << outData.submeshes.size() << " submeshes, "
                  << outData.materials.size() << " materials).\n";
        
        return true;
    }

    Engine::Scene::RenderableEntity loadModel(const std::string &filepath)
    {
        LoadedMeshData rawData;
        if (!loadOBJ(filepath, rawData))
        {
            throw std::runtime_error("ModelLoader failed to load OBJ file: " + filepath);
        }

        Engine::Scene::RenderableEntity entity{};

        // 1. Convert raw positions, normals, and texcoords into Engine::Scene::Vertex array
        size_t vertexCount = rawData.positions.size() / 3;
        entity.meshData.vertices.reserve(vertexCount);

        for (size_t i = 0; i < vertexCount; ++i)
        {
            Engine::Scene::Vertex v{};
            v.position = {rawData.positions[3 * i + 0], rawData.positions[3 * i + 1], rawData.positions[3 * i + 2]};

            if (!rawData.normals.empty())
            {
                v.normal = {rawData.normals[3 * i + 0], rawData.normals[3 * i + 1], rawData.normals[3 * i + 2]};
            }

            if (!rawData.texCoords.empty())
            {
                v.texCoord = {rawData.texCoords[2 * i + 0], rawData.texCoords[2 * i + 1]};
            }

            entity.meshData.vertices.push_back(v);
        }

        // 2. Copy indices
        entity.meshData.indices = rawData.indices;

        // 3. Convert raw submeshes
        entity.meshData.submeshes.reserve(rawData.submeshes.size());
        for (const auto &rawSub : rawData.submeshes)
        {
            Engine::Scene::Submesh sm{};
            sm.indexCount = rawSub.indexCount;
            sm.firstIndex = rawSub.firstIndex;
            sm.materialIndex = rawSub.materialIndex;
            entity.meshData.submeshes.push_back(sm);
        }

        // 4. Convert raw materials
        entity.materials.reserve(rawData.materials.size());
        for (const auto &rawMat : rawData.materials)
        {
            Engine::Scene::Material mat{};
            // Combine base directory with the texture filename to form the full path
            if (!rawMat.diffuseTexName.empty())
            {
                mat.diffuseTexPath = rawData.baseDir + rawMat.diffuseTexName;
            }
            mat.diffuseColor = rawMat.diffuseColor;
            // Default roughness or standard fallback can be assigned here
            mat.roughness = 0.5f;

            entity.materials.push_back(mat);
        }

        std::cout << "Successfully converted raw asset data into CPU Scene Entity ("
                  << vertexCount << " vertices, "
                  << entity.meshData.submeshes.size() << " submeshes).\n";

        return entity;
    }

} // namespace Utils