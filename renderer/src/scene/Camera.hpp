#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
/**
*
 * @brief CameraState struct represents the state of a camera in 3D space.
 *
 * This struct contains the position, target, up vector, field of view (FOV),
 * near and far clipping planes of the camera. It also provides methods to
 * compute the view and projection matrices based on the current state.
 *
 * @note The projection matrix is adjusted to account for Vulkan's coordinate system.
*/
namespace engine {
    struct CameraState {
        glm::vec3 position{0.0f};
        glm::vec3 target{0.0f, 0.0f, -1.0f};
        glm::vec3 up{0.0f, 1.0f, 0.0f};
        float fovYdeg = 60.0f;
        float zNear = 0.1f;
        float zFar = 1000.0f;

        glm::mat4 view() const {return glm::lookAt(position, target, up);}
        glm::mat4 projection(float aspect) const {
            glm::mat4 p = glm::perspective(glm::radians(fovYdeg), aspect, zNear, zFar);
            p[1][1] *= -1;
            return p;
        }
    };

    
}

