#pragma once

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace Engine::Scene
{
    /**
     * @brief Represents a standard 3D perspective camera for the rendering pipeline.
     * Manages view and projection transformations.
     */
    class Camera
    {
    public:
        /// Creates a camera with a given viewport aspect ratio (width / height).
        Camera(float aspectWidth, float aspectHeight);

        void setPosition(const glm::vec3 &position);
        void setOrientation(const glm::quat &orientation);

        /**
         * @brief Sets the rotation using Euler angles.
         * @param pitchDeg Pitch angle in degrees (rotation around X axis).
         * @param yawDeg   Yaw angle in degrees (rotation around Y axis).
         * @param rollDeg  Roll angle in degrees (rotation around Z axis).
         */
        void setRotationEuler(float pitchDeg, float yawDeg, float rollDeg);

        glm::mat4 getViewMatrix() const;
        glm::mat4 getProjectionMatrix() const;

        glm::vec3 getPosition() const { return m_position; }
        glm::quat getOrientation() const { return m_orientation; }

    private:
        glm::vec3 m_position{0.0f, 0.0f, 0.0f};
        glm::quat m_orientation{1.0f, 0.0f, 0.0f, 0.0f};

        float m_aspectRatio;
        float m_fovDegrees = 60.0f;
    };
} // namespace Engine::Scene