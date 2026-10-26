#version 450

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

// Descriptor binding for the material's texture sampler
layout(set = 0, binding = 1) uniform sampler2D texSampler;

void main() {
    // Sample the texture using the UV coordinates
    vec4 texColor = texture(texSampler, fragTexCoord);
    
    // Output the textured color (with an alpha check or fallback if needed)
    outColor = texColor;
}