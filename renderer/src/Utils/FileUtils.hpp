#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <iostream>

namespace Engine::Utils
{
    inline std::vector<char> readFile(const std::string &filename)
    {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        if (!file.is_open())
        {
            throw std::runtime_error("Failed to open file: " + filename);
        }

        size_t fileSize = file.tellg();
        std::vector<char> buffer(static_cast<size_t>(fileSize));

        file.seekg(0, std::ios::beg);
        file.read(buffer.data(), fileSize);

        return buffer;
    }
} // namespace Utils