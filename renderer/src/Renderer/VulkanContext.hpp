#pragma once
#define VULKAN_HPP_NO_CONSTRUCTORS

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include "../Core/Window.hpp"

class VulkanContext
{
public:
    VulkanContext(const Window &window);
    ~VulkanContext();

    VkDevice getDevice() const { return m_device; }
    VkPhysicalDevice getPhysicalDevice() const { return m_physicalDevice; }
    VmaAllocator getAllocator() const { return m_allocator; }
    VkQueue getGraphicsQueue() const { return m_graphicsQueue; }
    uint32_t getGraphicsQueueFamily() const { return m_graphicsQueueFamily; }

private:
    void createInstance();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void initVMA();

    VkInstance m_instance = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;
    VkQueue m_graphicsQueue = VK_NULL_HANDLE;

    uint32_t m_graphicsQueueFamily = 0;
    VmaAllocator m_allocator = VK_NULL_HANDLE;

    bool m_enableValidationLayers = true;
};