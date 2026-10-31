#pragma once

#include <vector>
#include <string>
#include <cstdint>

namespace Engine::Scene
{
    struct Texture
    {
        std::string name;       // Useful for debugging or asset identification
        std::string filepath;   // Source path on disk
        
        int32_t width{0};
        int32_t height{0};
        int32_t channels{0};
        
        std::vector<unsigned char> pixels;
        bool loadedSuccessfully{false};

        // Convenience helper
        bool isValid() const { return loadedSuccessfully && !pixels.empty(); }
    };
} // namespace Engine::Scene