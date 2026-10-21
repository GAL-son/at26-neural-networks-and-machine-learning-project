#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <iostream>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

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
        for (const auto &shape : shapes)
        {
            for (const auto &index : shape.mesh.indices)
            {
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
                    outData.normals.push_back(0.0f);
                    outData.normals.push_back(1.0f);
                    outData.normals.push_back(0.0f);
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
        }

        std::cout << "Successfully loaded OBJ: " << filepath << " (" << outData.indices.size() << " indices loaded).\n";
        return true;
    }

} // namespace Utils