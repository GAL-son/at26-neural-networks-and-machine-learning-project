#include "HeadlessConfig.hpp"

#include <fstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "Utils/FileUtils.hpp"

namespace Engine
{
    namespace
    {
        std::filesystem::path Resolve(const std::filesystem::path &base, const std::string &value)
        {
            const std::filesystem::path path = std::filesystem::u8path(value);
            return path.is_absolute() ? path : base / path;
        }

    }

    HeadlessConfig HeadlessConfig::Load(const std::filesystem::path &file)
    {
        std::ifstream stream(file);
        if (!stream)
            throw std::runtime_error("Cannot open config file: " + file.string());

        const std::filesystem::path baseDir = file.parent_path();
        HeadlessConfig config;

        try
        {
            const nlohmann::json json = nlohmann::json::parse(stream);

            config.scenesDir = Resolve(baseDir, json.at("scenesDir").get<std::string>());
            config.outputDir = Resolve(baseDir, json.value<std::string>("outputDir", "captures"));

            if (json.contains("scenes"))
                for (const auto &scene : json.at("scenes"))
                    config.scenes.push_back(Resolve(config.scenesDir, scene.get<std::string>()));
        }
        catch (const nlohmann::json::exception &e)
        {
            throw std::runtime_error(file.string() + ": " + e.what());
        }

        if (config.scenes.empty())
            config.scenes = Utils::FindObjFiles(config.scenesDir); // No list -> record everything

        if (config.scenes.empty())
            throw std::runtime_error("No scenes to record");

        // Fail fast
        std::string missing;
        for (const auto &scene : config.scenes)
            if (!std::filesystem::is_regular_file(scene))
                missing += "\n " + scene.string();

        if (!missing.empty())
            throw std::runtime_error("Missing scenes: " + missing);

        return config;
    }
}
