#version 450

layout(location = 0) in vec2 inUV;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inMotionVector;

layout(location = 0) out vec4 outColor;          
layout(location = 1) out vec4 outNormal;         
layout(location = 2) out vec4 outMotionVectors;    

layout(binding = 0) uniform MaterialUBO {
    vec3 ambient;
    float shininess;
    vec3 diffuse;
    float opacity;
    vec3 specular;
    int hasDiffuseTexture;
} material;

layout(binding = 1) uniform sampler2D texSampler;

void main() {
    vec4 baseColor;
    if (material.hasDiffuseTexture != 0) {
        baseColor = texture(texSampler, inUV);
    } else {
        baseColor = vec4(material.diffuse, material.opacity);
    }

    outColor         = baseColor;
    outNormal        = vec4(normalize(inNormal) * 0.5f + 0.5f, 1.0f);
    outMotionVectors = vec4(inMotionVector, 0.0f, 1.0f); // Prawidłowe wektory ruchu
}