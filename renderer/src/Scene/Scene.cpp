#include "Scene.hpp"

#define TINYOBJLOADER_IMPLEMENTATION

#include <tiny_obj_loader.h>
#include <stdexcept>
#include <iostream>
#include <unordered_map>

#include "../Utils/FileUtils.hpp"

Scene::Scene(VmaAllocator allocator, float aspectWidth, float aspectHeight)
    : m_allocator(allocator), m_camera(aspectWidth, aspectHeight)
{
    // Model center is roughly (8.5, -4.8, 21.4)
    // Place camera further out on the Z-axis (at Z = 90.0) looking directly at the center
    m_camera.setPosition(glm::vec3(8.5f, -4.8f, 90.0f));

    // Look straight ahead down the -Z axis into the bedroom
    m_camera.setRotationEuler(0.0f, 0.0f, 0.0f);
}

void Scene::loadModel(const std::string &filepath)
{
    Utils::LoadedMeshData rawData;
    if (Utils::loadOBJ(filepath, rawData))
    {
        m_meshes.push_back(Mesh::createFromLoadedData(m_allocator, rawData));
        std::cout << "Successfully uploaded bedroom mesh to GPU via VMA!\n";
    }
}

void Scene::render(VkCommandBuffer commandBuffer) const
{
    for (const auto &mesh : m_meshes)
    {
        mesh->draw(commandBuffer);
    }
}