// Shared albedo shading 

layout (set = 0, binding = 0) uniform MaterialUBO {
    vec3 ambient;
    float shininess;
    vec3 diffuse;
    float opacity;
    vec3 specular;
    int hasDiffuseTexture;
} material;

layout (set = 0, binding = 1) uniform sampler2D texSampler;

vec4 shadeAlbedo(vec2 uv) {
    if(material.hasDiffuseTexture != 0) {
        return texture(texSampler,uv);
    }
    return vec4(material.diffuse, material.opacity);
}