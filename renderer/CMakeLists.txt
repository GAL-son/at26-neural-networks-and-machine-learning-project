cmake_minimum_required(VERSION 3.20)
project(at26-neural-networks-and-machine-learning-project-renderer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(Vulkan REQUIRED)

add_executable(${PROJECT_NAME} src/main.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE src)

target_link_libraries(${PROJECT_NAME} PRIVATE 
    glfw
    glm::glm
    Vulkan::Vulkan
)