#pragma once

#include <filesystem>
#include <vector>
#include <string>

namespace Engine {
    struct HeadlessConfig {
        std::filesystem::path scenesDir;
        std::vector<std::filesystem::path> scenes;
        std::filesystem::path outputDir;

        static HeadlessConfig Load(const std::filesystem::path &file);
    };
}