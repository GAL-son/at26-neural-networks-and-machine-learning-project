#version 450

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D texColor;        // Top-Left: Kolor
layout(set = 0, binding = 1) uniform sampler2D texNormals;     // Top-Right: Normalne
layout(set = 0, binding = 2) uniform sampler2D texMotion;      // Bottom-Left: Wektory ruchu
layout(set = 0, binding = 3) uniform sampler2D texDepth;       // Bottom-Right: Głębokość

void main() {
    vec2 uv = fragTexCoord;
    
    if (uv.x < 0.5 && uv.y > 0.5) {
        // Top-Left: Color
        outColor = texture(texColor, vec2(uv.x * 2.0, (uv.y - 0.5) * 2.0));
    } else if (uv.x >= 0.5 && uv.y > 0.5) {
        // Top-Right: Normals
        outColor = texture(texNormals, vec2((uv.x - 0.5) * 2.0, (uv.y - 0.5) * 2.0));
    } else if (uv.x < 0.5 && uv.y <= 0.5) {
        // Bottom-Left: Motion Vectors
        outColor = texture(texMotion, vec2(uv.x * 2.0, uv.y * 2.0));
    } else {
       float depth = texture(texDepth, vec2((uv.x - 0.5) * 2.0, uv.y * 2.0)).r;
        
        // Jeśli depth == 1.0, to jest to tło. Odwracamy to:
        // Tło stanie się czarne (0.0), a obiekty im bliżej, tym bielsze.
        float mappedDepth = 1.0 - depth;
        
        // Ponieważ obiekty mogą być bardzo blisko zakresu 1.0, mnożymy to przez współczynnik wzmocnienia (kontrastu).
        // Możesz dostosować tę wartość (np. * 5.0 lub * 20.0), żeby uwydatnić szczegóły.
        mappedDepth *= 10.0; 

        outColor = vec4(vec3(mappedDepth), 1.0);
    }
}