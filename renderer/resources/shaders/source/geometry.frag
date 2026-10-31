#version 450

// Dane wejściowe z vertex shadera
layout(location = 0) in vec2 inUV;
layout(location = 1) in vec3 inNormal;

// Wyjścia G-Buffera / MRT
layout(location = 0) out vec4 outColor;          
layout(location = 1) out vec4 outNormal;         
layout(location = 2) out vec4 outMotionVectors;  
layout(location = 3) out vec4 outGroundTruth;    

// Binding 0: Właściwości materiału (UBO)
layout(binding = 0) uniform MaterialUBO {
    vec3 ambient;
    float shininess;
    vec3 diffuse;
    float opacity;
    vec3 specular;
    int hasDiffuseTexture;
} material;

// Binding 1: Próbnik tekstury diffuse
layout(binding = 1) uniform sampler2D texSampler;

void main() {
    vec4 baseColor;
    
    if (material.hasDiffuseTexture != 0) {
        baseColor = texture(texSampler, inUV);
    } else {
        baseColor = vec4(material.diffuse, material.opacity);
    }

    // Wypełnienie buforów wyjściowych G-Buffera
    outColor         = baseColor;
    outNormal        = vec4(normalize(inNormal) * 0.5f + 0.5f, 1.0f);
    outMotionVectors = vec4(0.0f, 0.0f, 0.0f, 1.0f);
    outGroundTruth   = baseColor;
}