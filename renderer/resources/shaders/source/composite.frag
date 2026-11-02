#version 450

layout(location = 0) in vec2 fragTexCoord; // (0,0) = top left
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D texColor;    
layout(set = 0, binding = 1) uniform sampler2D texNormals;    
layout(set = 0, binding = 2) uniform sampler2D texMotion;      
layout(set = 0, binding = 3) uniform sampler2D texDepth;

layout(push_constant) uniform CompositeParams {
    float mvScale; // Motion vector visual gain
    float nearPlane;
    float farPlane;
    float depthRange;
} params;

// Helper function to linearize depth values (adjust for your projection matrix if needed)
float linearizeDepth(float depth) {
    return params.nearPlane * params.farPlane / (params.farPlane - depth * (params.farPlane - params.nearPlane));
}

void main() {
    vec2 uv = fragTexCoord;
    bool left = uv.x < 0.5;
    bool top = uv.y < 0.5; // Fixed: Changed uv.x to uv.y for vertical split

    vec2 local = (uv - vec2(left ? 0.0 : 0.5, top ? 0.0 : 0.5)) * 2.0;

    if(top && left) { // Albedo 
        outColor = texture(texColor, local);
    } else if (top) { // Normal
        vec4 n = texture(texNormals, local);
        outColor = vec4(n.a > 0.0 ? n.xyz * 0.5 + 0.5 : vec3(0.0), 1.0);
    } else if (left) { // Motion Vector
        vec2 mv = texture(texMotion, local).rg;
        outColor = vec4(mv * params.mvScale * 0.5 + 0.5, 0.5, 1.0);
    } else { // Depth
        float z = linearizeDepth(texture(texDepth, local).r);
        outColor = vec4(vec3(1.0 - clamp(z / params.depthRange, 0.0, 1.0)), 1.0);
    }
}