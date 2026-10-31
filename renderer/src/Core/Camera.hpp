#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace Engine
{
    class Camera
    {
    public:
        Camera(float fov, float aspectRatio, float nearPlane, float farPlane);
        ~Camera() = default;

        // Ustawienia pozycji i orientacji
        void setPosition(const glm::vec3 &position)
        {
            m_position = position;
            updateViewMatrix();
        }
        void setRotation(float pitch, float yaw)
        {
            m_pitch = pitch;
            m_yaw = yaw;
            updateViewMatrix();
        }
        void translate(const glm::vec3 &delta)
        {
            m_position += delta;
            updateViewMatrix();
        }

        // Gettery macierzy dla shaderów (Zwracamy przez wartość, aby uniknąć Undefined Behavior!)
        const glm::mat4 &getViewMatrix() const { return m_viewMatrix; }
        const glm::mat4 &getProjectionMatrix() const { return m_projectionMatrix; }
        
        glm::mat4 getVPMatrix() const { return m_projectionMatrix * m_viewMatrix; } // POPRAWIONA KOLEJNOŚĆ I BRAK REFERENCJI DO TYMCZASOWEGO OBIEKTU
        const glm::mat4 &getPreviousVpMatrix() const { return m_previousVp; }

        // Obsługa zmiany rozmiaru okna (resize)
        void updateAspectRatio(float aspectRatio);
        
        void updatePreviousVp() 
        { 
            m_previousVp = m_projectionMatrix * m_viewMatrix; // Poprawna kolejność
        }

    private:
        void updateViewMatrix();

        glm::vec3 m_position{0.0f, 0.0f, 3.0f};
        glm::vec3 m_front{0.0f, 0.0f, -1.0f};
        glm::vec3 m_up{0.0f, 1.0f, 0.0f};
        glm::vec3 m_right{1.0f, 0.0f, 0.0f};

        float m_pitch{0.0f};
        float m_yaw{-90.0f};

        float m_fov;
        float m_aspectRatio;
        float m_nearPlane;
        float m_farPlane;

        glm::mat4 m_viewMatrix{1.0f};
        glm::mat4 m_projectionMatrix{1.0f};
        glm::mat4 m_previousVp{1.0f};
    };
}