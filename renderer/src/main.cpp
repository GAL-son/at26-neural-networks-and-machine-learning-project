#include "Core/Engine.hpp"
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    bool headless = true; 

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--visualize" || arg == "-v") {
            headless = false;
        }
    }

    try {
        Engine engine(headless);
        engine.run();
    } catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << "\n";
        return -1;
    }

    return 0;
}