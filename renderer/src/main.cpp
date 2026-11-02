#include "Core/Engine.hpp"
#include <iostream>
#include <exception>

int main(int argc, char *argv[])
{
    Engine::LaunchOptions options;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if(arg == "--headless")
            options.headless = true;

        else if (arg == "--config" && i + 1 < argc)
            options.configPath = argv[++i];
        else {
            std::cerr << "Unknown argument: " << arg << std::endl << "Usage: [--headless] [--config path/to/config.json]" << std::endl;
        }
    }
    
    try
    {
        Engine::Engine engine(options); // or whatever flag
        return engine.run();
    }
    catch (const std::exception &e)
    {
        std::cerr << "FATAL EXCEPTION: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}