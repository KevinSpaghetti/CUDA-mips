#include "Image.cu"
#include <vector>


std::vector<Image> generateImageContainersForMipMaps(const Image& original){
    //Dimensioni a cui dobbiamo fermarci
    const int minImageSizeX = 32;
    const int minImageSizeY = 32;
    
    std::vector<Image> result;
    result.push_back(original);

    float2 potImageDimensions = make_float2(float(original.width()), float(original.height()));

    int2 containerDimensions = make_int2(potImageDimensions.x / 2, potImageDimensions.y / 2);
    do{
        result.emplace_back(containerDimensions.x, containerDimensions.y, original.channels());
        containerDimensions.x /= 2;
        containerDimensions.y /= 2;
    
    }while(containerDimensions.x >= minImageSizeX && containerDimensions.y >= minImageSizeY);
    

    return result;
}
