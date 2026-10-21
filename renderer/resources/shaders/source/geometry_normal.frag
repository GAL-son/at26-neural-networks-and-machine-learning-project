#version 450

layout(location = 0) in vec3 fragNormal;
layout(location = 0) out vec4 outColor;

void main() {
    // Map normal from [-1, 1] to [0, 1] for RGB visualization
    vec3 normalizedNormal = normalize(fragNormal);
    vec3 color = normalizedNormal * 0.5f + 0.5f;
    
    outColor = vec4(color, 1.0f);
}