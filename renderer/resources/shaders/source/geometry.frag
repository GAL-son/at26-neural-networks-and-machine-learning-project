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

    // sun
    vec3 sunDir   = normalize(vec3(0.5, 1.0, 0.3));
    vec3 sunColor = vec3(1.0, 0.95, 0.8) * 0.8;    
    vec3 ambient  = vec3(0.2);                      

    vec3 norm = normalize(inNormal);
    float NdotL = max(dot(norm, sunDir), 0.0);
    vec3 lighting = ambient + sunColor * NdotL;
    // -------------------------------------

    // Output targets
    vec4 albedo = shadeAlbedo(inUV);
    outColor           = vec4(albedo.rgb * lighting, albedo.a); 
    outNormal          = vec4(norm, 1.0f);
    outMotionVectors   = vec4((curr - prev) * 0.5, 0.0, 1.0);
}