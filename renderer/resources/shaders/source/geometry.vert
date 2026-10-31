#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec2 outUV;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec2 outMotionVector;

layout(push_constant) uniform PushConsts {
    mat4 vp;
    mat4 prevVp;
} pushConsts;

void main() {
    outUV = inTexCoord;
    outNormal = inNormal;

    vec4 currentPos = pushConsts.vp * vec4(inPosition, 1.0);
    vec4 prevPos = pushConsts.prevVp * vec4(inPosition, 1.0);

    // Pozycje w przestrzeni ekranu (NDC)
    vec2 currentNDC = currentPos.xy / currentPos.w;
    vec2 prevNDC = prevPos.xy / prevPos.w;

    // Obliczenie wektora ruchu (różnica pozycji)
    outMotionVector = (currentNDC - prevNDC) * 0.5f;

    gl_Position = currentPos;
}