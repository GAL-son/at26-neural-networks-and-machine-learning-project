#include <vector>
#include <string>

namespace Utils::Images
{
    struct Image
    {
        int width = 0;
        int height = 0;
        int channels = 0;
        std::vector<unsigned char> pixels;
        std::string filepath;
        bool loadedSuccessfully = false;
    };
} // namespace Utils::Images