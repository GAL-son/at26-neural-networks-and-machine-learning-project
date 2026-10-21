#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <tiny_obj_loader.h>

#include<glm/glm.hpp>

namespace Utils
{

    inline std::vector<char> readFile(const std::string &filename)
    {
        // std::ios::ate starts the read pointer at the end of the file (great for getting size)
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open())
        {
            throw std::runtime_error("Failed to open file: " + filename);
        }

        size_t fileSize = file.tellg();
        std::vector<char> buffer(static_cast<size_t>(fileSize));

        file.seekg(0, std::ios::beg);
        file.read(buffer.data(), fileSize);

        return buffer;
    }

    // Simple raw mesh structure for loading
    struct LoadedMeshData
    {
        std::vector<float> positions;
        std::vector<float> normals;
        std::vector<float> texCoords;
        std::vector<uint32_t> indices;
    };

    inline bool loadOBJ(const std::string &filepath, LoadedMeshData &outData)
    {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;

        // Extract base directory for material/texture loading
        std::string baseDir = filepath.substr(0, filepath.find_last_of("/\\") + 1);

        bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filepath.c_str(), baseDir.c_str());

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

        // Loop over shapes and flatten into a single mesh or collection
        // Loop over shapes and flatten into a single mesh or collection
        for (const auto &shape : shapes)
        {
            size_t index_offset = 0;
            for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++)
            {
                int fv = shape.mesh.num_face_vertices[f];

                // Collect the 3 vertices of the triangle to compute a face normal if needed
                tinyobj::index_t idx0 = shape.mesh.indices[index_offset + 0];
                tinyobj::index_t idx1 = shape.mesh.indices[index_offset + 1];
                tinyobj::index_t idx2 = shape.mesh.indices[index_offset + 2];

                glm::vec3 v0(attrib.vertices[3 * idx0.vertex_index + 0], attrib.vertices[3 * idx0.vertex_index + 1], attrib.vertices[3 * idx0.vertex_index + 2]);
                glm::vec3 v1(attrib.vertices[3 * idx1.vertex_index + 0], attrib.vertices[3 * idx1.vertex_index + 1], attrib.vertices[3 * idx1.vertex_index + 2]);
                glm::vec3 v2(attrib.vertices[3 * idx2.vertex_index + 0], attrib.vertices[3 * idx2.vertex_index + 1], attrib.vertices[3 * idx2.vertex_index + 2]);

                // Fallback flat normal calculated from triangle geometry
                glm::vec3 faceNormal = glm::normalize(glm::cross(v1 - v0, v2 - v0));

                for (int v = 0; v < fv; v++)
                {
                    tinyobj::index_t index = shape.mesh.indices[index_offset + v];

                    // Position
                    outData.positions.push_back(attrib.vertices[3 * index.vertex_index + 0]);
                    outData.positions.push_back(attrib.vertices[3 * index.vertex_index + 1]);
                    outData.positions.push_back(attrib.vertices[3 * index.vertex_index + 2]);

                    // Normal (Use file normal if available, otherwise use computed face normal)
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

                    outData.indices.push_back(static_cast<uint32_t>(outData.indices.size()));
                }
                index_offset += fv;
            }
        }

        std::cout << "Successfully loaded OBJ: " << filepath << " (" << outData.indices.size() << " indices loaded).\n";
        return true;
    }

} // namespace Utils