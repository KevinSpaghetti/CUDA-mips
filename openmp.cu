#include <omp.h>
#include <iostream>
#include <cmath>
#include <chrono>
#include "libs\times.cpp"
#include "libs\ImageUtils.cu"
#include "libs\operators.cu"
#include "libs\mipchain.cu"

void GenerateMipMap(const float4 *inputImage, float4 *outputImage, const int2 isize, const int2 osize){
    const int fwidth = 2;
    const int fsize = (fwidth * fwidth);
    const float fweight = 1.0 / (float) fsize;

    #pragma omp parallel for
    for(int i = 0; i < osize.x * osize.y; i++){     
        
        int y = i / osize.x;
        int x = i % osize.x;

        //Dobbiamo invertire la condizione perchè in OpenMP
        //non è permesso avere un return dentro una sezione parallela
        if(y < osize.y || x < osize.x) {
            float4 result = make_float4(0, 0, 0, 0);
            for(int row = 0; row < fwidth; ++row){
                for(int column = 0; column < fwidth; ++column){
                    int2 boxOffset = make_int2(fwidth * x, fwidth * y);
                    int2 boxCoords = make_int2(boxOffset.x + row, boxOffset.y + column);

                    boxCoords.x = max(boxCoords.x, 0);
                    boxCoords.y = max(boxCoords.y, 0);
                    boxCoords.x = min(boxCoords.x, isize.x - 1);
                    boxCoords.y = min(boxCoords.y, isize.y - 1);
        
                    result = result + inputImage[tolinear(boxCoords, isize.x)];
                }
            }   
            outputImage[tolinear(make_int2(x, y), osize.x)] = result * fweight;
        }
    }


}

void generateMipMapChain(const std::string folder, const std::string destination, const std::string filename, times& measurements){

    const std::string filepath = folder + filename;

    auto start = std::chrono::steady_clock::now();
    auto original = ImageUtils::load(filepath.c_str());
    auto end = std::chrono::steady_clock::now();
    measurements.image_reading = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    measurements.image_reading /= 1000;

    auto containers = generateImageContainersForMipMaps(original);
    
    start = std::chrono::steady_clock::now();
    for(int i = 0; i < containers.size() - 1; ++i){
        int2 isize = make_int2(containers[i].width(), containers[i].height());
        int2 osize = make_int2(containers[i + 1].width(), containers[i + 1].height());
        GenerateMipMap((float4*) containers[i].raw_data(), (float4*) containers[i + 1].raw_data(), isize, osize);
    }
    end = std::chrono::steady_clock::now();
    measurements.image_processing = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    measurements.image_processing /= 1000;

    start = std::chrono::steady_clock::now();
    int writer_threads = containers.size() - 1;
    #pragma omp parallel for num_threads(writer_threads)
    for(int i = 1; i < containers.size(); i++){
        std::string save_dest = destination + "mip_" + std::to_string(containers[i].width()) + "x" + std::to_string(containers[i].height()) + "_" + filename;
        ImageUtils::save(containers[i], save_dest.c_str());
    }
    end = std::chrono::steady_clock::now();
    measurements.image_writing = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count(); 
    measurements.image_writing /= 1000;
}

int main(int argc, char** argv){

    const std::string filepath = std::string(argv[1]) + std::string(argv[3]);
    
    auto img = ImageUtils::load(filepath.c_str());

    std::cout << "algorithm" << "," << "image" << "," << "width" << "," << "height" << "," << "reading" << "," << "processing" << "," << "writing" << "\n";
    int runs = 10;
    for(int run = 1; run <= runs; run++){        
        times times;
        generateMipMapChain(std::string(argv[1]), std::string(argv[2]), std::string(argv[3]), times);
        std::cout 
        << "OpenMP" << ", " 
        << argv[3] << " , " 
        << img.width() << " , " << img.height() << " , " 
        << times.image_reading << ", " 
        << times.image_processing << ", "
        << times.image_writing << ""<< "\n";
    }


    return 0;
}
