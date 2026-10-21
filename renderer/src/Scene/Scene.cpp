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
    // Position camera at origin looking into the bedroom scene
    m_camera.setPosition(glm::vec3(0.0f, 1.5f, 3.0f));
}

void Scene::loadModel(const std::string& filepath) {
    Utils::LoadedMeshData rawData;
    if (Utils::loadOBJ(filepath, rawData)) {
        m_meshes.push_back(Mesh::createFromLoadedData(m_allocator, rawData));
        std::cout << "Successfully uploaded bedroom mesh to GPU via VMA!\n";
    }
}

void Scene::render(VkCommandBuffer commandBuffer) const {
    for (const auto& mesh : m_meshes) {
        mesh->draw(commandBuffer);
    }
}