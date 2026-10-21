#include "Camera.hpp"

Camera::Camera(float aspectWidth, float aspectHeight) : m_aspectRatio(aspectWidth / aspectHeight)
{
}

void Camera::setPosition(const glm::vec3& position) {
    m_position = position;
}

void Camera::setOrientation(const glm::quat& orientation) {
    m_orientation = glm::normalize(orientation);
}

void Camera::setRotationEuler(float pitchDeg, float yawDeg, float rollDeg) {
    glm::vec3 radiansEuler = glm::radians(glm::vec3(pitchDeg, yawDeg, rollDeg));
    
    // GLM euler angle order (Pitch = X, Yaw = Y, Roll = Z)
    glm::quat qPitch = glm::angleAxis(radiansEuler.x, glm::vec3(1.0f, 0.0f, 0.0f));
    glm::quat qYaw   = glm::angleAxis(radiansEuler.y, glm::vec3(0.0f, 1.0f, 0.0f));
    glm::quat qRoll  = glm::angleAxis(radiansEuler.z, glm::vec3(0.0f, 0.0f, 1.0f));

    m_orientation = glm::normalize(qYaw * qPitch * qRoll);
}

glm::mat4 Camera::getViewMatrix() const {
    glm::vec3 forward = m_orientation * glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 up = m_orientation * glm::vec3(0.0f, 1.0f, 0.0f);
    
    return glm::lookAt(m_position, m_position + forward, up);
}

glm::mat4 Camera::getProjectionMatrix() const {
    glm::mat4 proj = glm::perspective(glm::radians(m_fov), m_aspectRatio, 0.1f, 100.0f);
    proj[1][1] *= -1; // Flip Y for Vulkan coordinate system convention
    return proj;
}