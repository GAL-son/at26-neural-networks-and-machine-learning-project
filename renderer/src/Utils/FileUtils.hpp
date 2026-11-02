#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <filesystem>

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

    inline std::vector<std::filesystem::path> FindObjFiles(const std::filesystem::path &dir)
    {
        std::vector<std::filesystem::path> result;
        std::error_code ec;

        for (
            auto it = std::filesystem::recursive_directory_iterator(dir, std::filesystem::directory_options::skip_permission_denied, ec);
            !ec && it != std::filesystem::recursive_directory_iterator();
            it.increment(ec))
        {
            if(!it->is_regular_file(ec))
                continue;

            std::string ext = it->path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {return char(std::tolower(c)); });
            if(ext == ".obj")
                result.push_back(it->path());
        }

        std::sort(result.begin(), result.end());
        return result;
    }

} // namespace Utils