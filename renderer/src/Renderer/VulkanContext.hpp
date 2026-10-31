#pragma once
#define VULKAN_HPP_NO_CONSTRUCTORS

#include "../Utils/Macros.hpp"

#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.h>
#include "../Core/Window.hpp"
#include <vector>

class VulkanContext
{
public:
    explicit VulkanContext(const Window &window);
    ~VulkanContext();

    DISALLOW_COPY(VulkanContext)
    DISALLOW_MOVE(VulkanContext)

    // Getters
    VkDevice getDevice() const { return m_device; }
    VkInstance getInstance() const { return m_instance; }
    VkPhysicalDevice getPhysicalDevice() const { return m_physicalDevice; }
    VmaAllocator getAllocator() const { return m_allocator; }
    VkQueue getGraphicsQueue() const { return m_graphicsQueue; }
    uint32_t getGraphicsQueueFamily() const { return m_graphicsQueueFamily; }

    // Swapchain accessors for rendering
    VkSwapchainKHR getSwapchain() const { return m_swapchain; }
    VkFormat getSwapchainImageFormat() const { return m_swapchainImageFormat; }
    const std::vector<VkImageView> &getSwapchainImageViews() const { return m_swapchainImageViews; }
    const std::vector<VkImage> &getSwapchainImages() const { return m_swapchainImages; }
    VkExtent2D getSwapchainExtent() const { return m_swapchainExtent; }

    void recreateSwapchain(uint32_t width, uint32_t height);

private:
    // Vulkan initialization helpers
    void createInstance(const Window &window);
    void pickPhysicalDevice();
    void createLogicalDevice();
    void initVMA();

    // Vulkan render and display helpers
    void createSurface(const Window &window);
    void createSwapchain(int width, int height);
    void createImageViews();

    // Member variables
    const Window &m_window;

    VkInstance m_instance = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    bool m_enableValidationLayers = true;

    VkQueue m_graphicsQueue = VK_NULL_HANDLE;
    uint32_t m_graphicsQueueFamily = 0;
    VkQueue m_presentQueue = VK_NULL_HANDLE;
    uint32_t m_presentQueueFamily = 0;
    VkSurfaceKHR m_surface = VK_NULL_HANDLE;

    // Swapchain members
    VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;
    std::vector<VkImage> m_swapchainImages;
    std::vector<VkImageView> m_swapchainImageViews;
    VkFormat m_swapchainImageFormat = VK_FORMAT_B8G8R8A8_SRGB;
    VkExtent2D m_swapchainExtent{0, 0};
};