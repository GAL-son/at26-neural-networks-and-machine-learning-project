#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec2 outUV;
layout(location = 1) out vec3 outNormal;

layout(push_constant) uniform PushConsts {
    mat4 vp;
} pc;

void main() {
    outUV = inTexCoord;
    outNormal = inNormal;
    gl_Position = pc.vp * vec4(inPosition, 1.0);
}