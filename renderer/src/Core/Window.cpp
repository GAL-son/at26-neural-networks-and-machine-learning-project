#include "Window.hpp"
#include <stdexcept>

Window::Window(int width, int height, const std::string &title, bool headless) : m_width(width), m_height(height), m_headless(headless)
{
    if(m_headless) return;

    if(!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW!");
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    m_window = glfwCreateWindow(m_width, m_height, title.c_str(), nullptr, nullptr);
    if(!m_window) {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }
}

Window::~Window() {
    if(!m_headless && m_window) {
        glfwDestroyWindow(m_window);
        glfwTerminate();
    }
}

bool Window::shouldClose() const {
    if(m_headless) return false;
    return glfwWindowShouldClose(m_window);
}

void Window::poolEvents() const {
    if(!m_headless) {
        glfwPollEvents();
    }
}