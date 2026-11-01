#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec2 outUV;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec4 outCurrClip;
layout(location = 3) out vec4 outPrevClip;

layout(push_constant) uniform PushConsts {
    mat4 vp;
    mat4 prevVp;
} pushConsts;

void main() {
    outUV = inTexCoord;
    outNormal = inNormal;

    vec4 currentPos = pushConsts.vp * vec4(inPosition, 1.0);
    // vec4 prevPos = pushConsts.prevVp * vec4(inPosition, 1.0);

    // Obliczenie wektora ruchu (różnica pozycji)
    outCurrClip = currentPos;
    outPrevClip = pushConsts.prevVp * vec4(inPosition, 1.0);

    gl_Position = currentPos;
}