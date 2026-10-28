/**
 * @file Camera.hpp
 * @brief Header file for the Camera class in the Engine::Scene namespace.
 *
 * This file defines the Camera class, which is responsible for handling camera operations
 * in a 3D rendering engine. The Camera class manages the camera's position, orientation,
 * and provides methods to set these properties as well as to retrieve the view and projection matrices.
 */

#pragma once

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace Engine::Scene
{
    /**
     * @class Camera
     * @brief Represents a camera in the 3D scene.
     *
     * The Camera class provides methods to set and get the camera's position and orientation,
     * as well as to retrieve the view and projection matrices for rendering.
     */
    class Camera
    {
    public:
        /**
         * @brief Constructor for the Camera class.
         *
         * @param aspectWidth The width of the viewport's aspect ratio.
         * @param aspectHeight The height of the viewport's aspect ratio.
         */
        Camera(float aspectWidth, float aspectHeight);

        /**
         * @brief Sets the position of the camera.
         *
         * @param position The new position of the camera as a glm::vec3.
         */
        void setPosition(const glm::vec3 &position);

        /**
         * @brief Sets the orientation of the camera using a quaternion.
         *
         * @param orientation The new orientation of the camera as a glm::quat.
         */
        void setOrientation(const glm::quat &orientation);

        /**
         * @brief Sets the rotation of the camera using Euler angles.
         *
         * @param pitchDeg The pitch angle in degrees.
         * @param yawDeg The yaw angle in degrees.
         * @param rollDeg The roll angle in degrees.
         */
        void setRotationEuler(float pitchDeg, float yawDeg, float rollDeg);

        /**
         * @brief Retrieves the view matrix of the camera.
         *
         * @return The view matrix as a glm::mat4.
         */
        glm::mat4 getViewMatrix() const;

        /**
         * @brief Retrieves the projection matrix of the camera.
         *
         * @return The projection matrix as a glm::mat4.
         */
        glm::mat4 getProjectionMatrix() const;

        /**
         * @brief Retrieves the current position of the camera.
         *
         * @return The position of the camera as a glm::vec3.
         */
        glm::vec3 getPosition() const { return m_position; }

        /**
         * @brief Retrieves the current orientation of the camera.
         *
         * @return The orientation of the camera as a glm::quat.
         */
        glm::quat getOrientation() const { return m_orientation; }

    private:
        glm::vec3 m_position{0.0f, 0.0f, 0.0f}; /**< The position of the camera. */
        glm::quat m_orientation{1.0f, 0.0f, 0.0f, 0.0f}; /**< The orientation of the camera. */

        float m_aspectRatio; /**< The aspect ratio of the viewport. */
        float m_fov = 60; /**< The field of view of the camera in degrees. */
    };
} // namespace Engine::Scene
