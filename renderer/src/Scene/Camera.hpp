#pragma once

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

class Camera
{
public:
    Camera(float aspectWidth, float aspectHeight);

    void setPosition(const glm::vec3 &position);
    void setOrientation(const glm::quat &orientation);

    void setRotationEuler(float pitchDeg, float yawDeg, float rollDeg);

    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjectionMatrix() const;

    glm::vec3 getPosition() const { return m_position; }
    glm::quat getOrientation() const { return m_orientation; }

private:
    glm::vec3 m_position{0.0f, 0.0f, 0.0f};
    glm::quat m_orientation{1.0f, 0.0f, 0.0f, 0.0f};

    float m_aspectRatio;
    float m_fov = 60;
};