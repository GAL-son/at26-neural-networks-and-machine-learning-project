#pragma once

#include "Mesh.hpp"
#include "Camera.hpp"
#include <vector>
#include <memory>
#include <string>

class Scene {
public:
    Scene(VmaAllocator allocator, float aspectWidth, float aspectHeight);
    ~Scene() = default;

    void loadModel(const std::string& filepath);
    void render(VkCommandBuffer commandBuffer) const;

    Camera& getCamera() { return m_camera; }
    const Camera& getCamera() const { return m_camera; }

private:
    VmaAllocator m_allocator;
    Camera m_camera;
    std::vector<std::unique_ptr<Mesh>> m_meshes;
};