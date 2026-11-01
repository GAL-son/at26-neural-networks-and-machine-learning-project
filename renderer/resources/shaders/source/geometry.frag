#version 450

#extension GL_GOOGLE_include_directive : require
#include "material.glsl"

layout(location = 0) in vec2 inUV;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inCurrClip;
layout(location = 3) in vec4 inPrevClip;

layout(location = 0) out vec4 outColor;          
layout(location = 1) out vec4 outNormal;         
layout(location = 2) out vec4 outMotionVectors;    

void main() {
    // NDC positions
    vec2 curr = inCurrClip.xy / inCurrClip.w;
    vec2 prev = inPrevClip.xy / inPrevClip.w;

    // Output targets (fixed inUV casing)
    outColor           = shadeAlbedo(inUV); 
    outNormal          = vec4(normalize(inNormal) * 0.5f + 0.5f, 1.0f);
    outMotionVectors = vec4((curr - prev) * 0.5, 0.0, 1.0);
}