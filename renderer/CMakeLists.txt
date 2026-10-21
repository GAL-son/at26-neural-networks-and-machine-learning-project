cmake_minimum_required(VERSION 3.20)
project(at26-neural-networks-and-machine-learning-project-renderer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(OpenEXR CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)

set(TARGET_NAME AIAARendererApp)
add_executable(${TARGET_NAME} 
    src/main.cpp
    src/Core/Engine.cpp
    src/Core/Window.cpp
    src/Renderer/VulkanContext.cpp
    src/UI/UIManager.cpp
)

target_include_directories(${TARGET_NAME} PRIVATE
    src
)

target_link_libraries(${TARGET_NAME} PRIVATE
    Vulkan::Vulkan
    glfw
    imgui::imgui
    OpenEXR::OpenEXR
    nlohmann_json::nlohmann_json
    GPUOpen::VulkanMemoryAllocator
)

add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/resources/shaders/target"
        "$<TARGET_FILE_DIR:${TARGET_NAME}>/shaders"
    COMMENT "Copying compiled shaders to build destination..."
)