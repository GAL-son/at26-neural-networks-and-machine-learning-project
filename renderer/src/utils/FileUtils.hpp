#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>

namespace Utils {

    inline std::vector<char> readFile(const std::string& filename) {
        // std::ios::ate starts the read pointer at the end of the file (great for getting size)
        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + filename);
        }
        
        size_t fileSize = file.tellg();
        std::vector<char> buffer(static_cast<size_t>(fileSize));
        
        file.seekg(0, std::ios::beg);
        file.read(buffer.data(), fileSize);
        
        return buffer;
    }

} // namespace Utils