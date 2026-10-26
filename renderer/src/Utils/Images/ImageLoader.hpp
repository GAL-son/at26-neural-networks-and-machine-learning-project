#pragma once

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <iostream>

#include "Image.hpp"

namespace Utils::Images
{

    inline Image loadImage(const std::string& filepath) {
        Image data{};
        data.filepath = filepath;

        stbi_set_flip_vertically_on_load(true);
        unsigned char* rawPixels = stbi_load(filepath.c_str(), &data.width, &data.height, &data.channels, STBI_rgb_alpha);

        if (!rawPixels) {
            std::cerr << "Warning: Failed to load texture from disk: " << filepath << ". Using 1x1 fallback.\n";
            // Provide a 1x1 white pixel fallback
            data.width = 1;
            data.height = 1;
            data.channels = 4;
            data.pixels = {255, 255, 255, 255};
            data.loadedSuccessfully = false;
            return data;
        }

        size_t imageSize = static_cast<size_t>(data.width * data.height * 4);
        data.pixels.resize(imageSize);
        memcpy(data.pixels.data(), rawPixels, imageSize);

        stbi_image_free(rawPixels);
        data.loadedSuccessfully = true;
        
        std::cout << "Loaded image to CPU: " << filepath << " (" << data.width << "x" << data.height << ")\n";
        return data;
    }

} // Utils::images
