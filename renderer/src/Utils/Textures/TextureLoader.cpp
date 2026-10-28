#include "TextureLoader.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <iostream>
#include <cstring> // For memcpy

namespace Engine::Utils
{
    Scene::Texture TextureLoader::loadFromFile(const std::string& filepath)
    {
        Scene::Texture texture{};
        texture.filepath = filepath;

        size_t lastSlash = filepath.find_last_of("/\\");
        texture.name = (lastSlash == std::string::npos) ? filepath : filepath.substr(lastSlash + 1);

        stbi_set_flip_vertically_on_load(true);
        unsigned char* rawPixels = stbi_load(filepath.c_str(), &texture.width, &texture.height, &texture.channels, STBI_rgb_alpha);

        if (!rawPixels)
        {
            std::cerr << "Warning: Failed to load texture from disk: " << filepath << ". Using 1x1 fallback.\n";
            
            // Provide a 1x1 white pixel fallback so rendering doesn't crash
            texture.width = 1;
            texture.height = 1;
            texture.channels = 4;
            texture.pixels = {255, 255, 255, 255};
            texture.loadedSuccessfully = false;
            return texture;
        }

        size_t imageSize = static_cast<size_t>(texture.width * texture.height * 4);
        texture.pixels.resize(imageSize);
        std::memcpy(texture.pixels.data(), rawPixels, imageSize);

        stbi_image_free(rawPixels);
        texture.loadedSuccessfully = true;
        
        std::cout << "Loaded texture to CPU: " << texture.name << " (" << texture.width << "x" << texture.height << ")\n";
        return texture;
    }
} // namespace Engine::Scene