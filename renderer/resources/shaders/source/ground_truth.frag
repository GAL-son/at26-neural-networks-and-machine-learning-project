#version 450

#extension GL_GOOGLE_include_directive : require
#include "material.glsl"

layout(location = 0) in vec2 inUV;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec4 outColor;    

void main() {
    vec3 sunDir   = normalize(vec3(0.5, 1.0, 0.3));
    vec3 sunColor = vec3(1.0, 0.95, 0.8) * 2.0;
    vec3 ambient  = vec3(0.2);

    vec3 norm = normalize(inNormal);
    float NdotL = max(dot(norm, sunDir), 0.0);
    vec3 lighting = ambient + sunColor * NdotL;

    vec4 albedo = shadeAlbedo(inUV);
    outColor = vec4(albedo.rgb * lighting, albedo.a);
}