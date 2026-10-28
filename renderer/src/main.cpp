#include "Core/Engine.hpp"
#include <iostream>
#include <exception>

int main(int argc, char* argv[]) {
    bool headless = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-v" || std::string(argv[i]) == "--headless") {
            // Adjust based on how your flag is checked, or keep as is:
        }
    }
    // Check for -v or similar in your code...
    bool isHeadless = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-v") {
            isHeadless = true; // Wait, -v in your log was running windowed mode or headless?
        }
    }

    try {
        Engine::Engine engine(false); // or whatever flag
        engine.run();
    } catch (const std::exception& e) {
        std::cerr << "FATAL EXCEPTION: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}