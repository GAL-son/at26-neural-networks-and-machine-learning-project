#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <iostream>

int main() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW!" << std::endl;
        return -1;
    }

    if (!glfwVulkanSupported()) {
        std::cerr << "Vulkan is not supported on this system!" << std::endl;
        glfwTerminate();
        return -1;
    }

    std::cout << "GLFW and Vulkan initialized successfully!" << std::endl;

    glfwTerminate();
    return 0;
}