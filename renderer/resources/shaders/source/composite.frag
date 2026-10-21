#version 450
layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D texBuffer0; // G-Buffer
layout(set = 0, binding = 1) uniform sampler2D texBuffer1; // Depth/Normals
layout(set = 0, binding = 2) uniform sampler2D texBuffer2; // Motion Vectors
layout(set = 0, binding = 3) uniform sampler2D texBuffer3; // SSAA GT

void main() {
    vec2 uv = fragTexCoord;
    
    if (uv.x < 0.5 && uv.y > 0.5) {
        // Top-Left: G-Buffer
        outColor = texture(texBuffer0, vec2(uv.x * 2.0, (uv.y - 0.5) * 2.0));
    } else if (uv.x >= 0.5 && uv.y > 0.5) {
        // Top-Right: Depth/Normals
        outColor = texture(texBuffer1, vec2((uv.x - 0.5) * 2.0, (uv.y - 0.5) * 2.0));
    } else if (uv.x < 0.5 && uv.y <= 0.5) {
        // Bottom-Left: Motion Vectors
        outColor = texture(texBuffer2, vec2(uv.x * 2.0, uv.y * 2.0));
    } else {
        // Bottom-Right: SSAA Ground Truth
        outColor = texture(texBuffer3, vec2((uv.x - 0.5) * 2.0, uv.y * 2.0));
    }
}