#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <string>

class Window
{
public:
    Window(int width, int height, const std::string &title, bool headless);
    ~Window();

    bool shouldClose() const;
    void poolEvents() const;

    GLFWwindow *getGLFWWindow() const { return m_window; }

    bool isHeadless() const { return m_headless; }
    int getWidth() const { return m_width; }
    int getHeight() const { return m_height; }

private:
    GLFWwindow *m_window = nullptr;
    int m_width, m_height;
    bool m_headless;
};