#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include "Camera.hpp"

namespace Engine
{
    Camera::Camera(float fov, float aspectRatio, float nearPlane, float farPlane)
        : m_fov(fov), m_aspectRatio(aspectRatio), m_nearPlane(nearPlane), m_farPlane(farPlane)
    {
        // Tworzymy macierz perspektywy
        m_projectionMatrix = glm::perspective(glm::radians(m_fov), m_aspectRatio, m_nearPlane, m_farPlane);

        // --- KLUCZOWE DLA VULCANA: Odwrócenie osi Y w przestrzeni klipu ---
        m_projectionMatrix[1][1] *= -1.0f;

        updateViewMatrix();
    }

    void Camera::updateAspectRatio(float aspectRatio)
    {
        m_aspectRatio = aspectRatio;
        m_projectionMatrix = glm::perspective(glm::radians(m_fov), m_aspectRatio, m_nearPlane, m_farPlane);
        m_projectionMatrix[1][1] *= -1.0f; // Pamiętamy o odwróceniu Y w Vulkanie!
    }

    void Camera::updateViewMatrix()
    {
        // Przeliczanie kątów Euler'a (Pitch i Yaw) na wektor kierunku (m_front)
        glm::vec3 front;
        front.x = cos(glm::radians(m_yaw)) * cos(glm::radians(m_pitch));
        front.y = sin(glm::radians(m_pitch));
        front.z = sin(glm::radians(m_yaw)) * cos(glm::radians(m_pitch));
        m_front = glm::normalize(front);

        // Opcjonalnie wyznaczamy wektor w prawo i w górę
        m_right = glm::normalize(glm::cross(m_front, glm::vec3(0.0f, 1.0f, 0.0f)));
        m_up = glm::normalize(glm::cross(m_right, m_front));

        m_viewMatrix = glm::lookAt(m_position, m_position + m_front, m_up);
    }
}