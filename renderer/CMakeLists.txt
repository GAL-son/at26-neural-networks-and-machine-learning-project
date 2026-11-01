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

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "src/*.cpp")

set(TARGET_NAME AIAARendererApp)
add_executable(${TARGET_NAME} ${SOURCES})

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

find_program(GLSLC glslc HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin" REQUIRED)

set(SHADER_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/resources/shaders/source")
set(SHADER_BIN_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")

# Upewnij się, że katalog docelowy dla shaderów zawsze istnieje
file(MAKE_DIRECTORY "${SHADER_BIN_DIR}")

# POPRAWKA: Usunięto przecinki (wzorce oddzielamy spacjami)
file(GLOB SHADER_SOURCES CONFIGURE_DEPENDS 
    "${SHADER_SRC_DIR}/*.vert" 
    "${SHADER_SRC_DIR}/*.frag" 
    "${SHADER_SRC_DIR}/*.comp"
)

set(SPIRV_FILES)
foreach(SHADER_SRC ${SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER_SRC} NAME)
    set(SPIRV "${SHADER_BIN_DIR}/${SHADER_NAME}.spv")
    list(APPEND SPIRV_FILES ${SPIRV})
    add_custom_command(
        OUTPUT ${SPIRV}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_BIN_DIR}
        COMMAND ${GLSLC} --target-env=vulkan1.3 -MD -MF ${SPIRV}.d -o ${SPIRV} ${SHADER_SRC}
        DEPENDS ${SHADER_SRC}
        DEPFILE ${SPIRV}.d
        COMMENT "Compiling shader ${SHADER_NAME}"
        VERBATIM
    )
endforeach()

add_custom_target(Shaders DEPENDS ${SPIRV_FILES})
add_dependencies(${TARGET_NAME} Shaders)

add_custom_command(TARGET Shaders POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${SHADER_BIN_DIR} "$<TARGET_FILE_DIR:${TARGET_NAME}>/shaders"
    VERBATIM
)

# Copy scenes
add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/resources/scenes"
        "$<TARGET_FILE_DIR:${TARGET_NAME}>/scenes"
    COMMENT "Copying scenes to build destination..."
)