#version 450

// Atrybuty wierzchołka (zgodne z układem bufora w Twojej klasie Mesh)
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

// Dane przekazywane do fragment shadera
layout(location = 0) out vec2 fragUV;
layout(location = 1) out vec3 fragNormal;

// Push Constants – szybkie przekazanie macierzy VP z kamery (64 bajty)
layout(push_constant) uniform PushConstants {
    mat4 viewProj;
} pc;

void main() {
    fragUV = inUV;
    fragNormal = inNormal;
    
    // Transformacja pozycji wierzchołka do przestrzeni klipu
    gl_Position = pc.viewProj * vec4(inPosition, 1.0f);
}