#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <glm/glm.hpp>
#include <string>

struct Texture {
    VkImage image = VK_NULL_HANDLE;
    VkImageView imageView = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    std::string filepath;
};

struct Material {
    std::string name;
    glm::vec3 diffuseColor = glm::vec3(1.0f);
    int diffuseTextureIndex = -1; // Points to an index in a scene-wide texture vector
    float shininess = 0.0f;
};