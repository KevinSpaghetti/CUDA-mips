#include <iostream>
#include <cmath>
#include <chrono>
#include <algorithm>
#include "libs/times.cpp"
#include "libs/ImageUtils.cu"
#include "libs/mipchain.cu"
#include "libs/operators.cu"

void checkCudaError(){
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("GPUassert: %s\n", cudaGetErrorString(err));
    }
}

__global__ void GenerateMipMap(const float4 *inputImage, float4 *outputImage, const int2 isize, const int2 osize){
    
    const int fwidth = 2;
    const int fsize = (fwidth * fwidth);
    const float fweight = 1.0 / (float) fsize;

    //Calcolo delle coordinate del pixel assegnato a questo thread
    const int2 blockStartCoords = make_int2(blockIdx.x * blockDim.x, blockIdx.y * blockDim.y);
    const int2 pxCoords = make_int2(blockStartCoords.x + threadIdx.x, blockStartCoords.y + threadIdx.y);

    if(pxCoords.x >= osize.x || pxCoords.y >= osize.y) return;

    float4 result = make_float4(0, 0, 0, 0);
    for(int row = 0; row < fwidth; ++row){
        for(int column = 0; column < fwidth; ++column){
            //Trasformazione delle coordinate del pixel (nell'immagine di output)
            //in coordinate nell'immagine di input
            int2 boxOffset = make_int2(fwidth * pxCoords.x, fwidth * pxCoords.y);
            int2 boxCoords = make_int2(boxOffset.x + row, boxOffset.y + column);

            boxCoords.x = max(boxCoords.x, 0);
            boxCoords.y = max(boxCoords.y, 0);
            boxCoords.x = min(boxCoords.x, isize.x - 1);
            boxCoords.y = min(boxCoords.y, isize.y - 1);

            result = result + inputImage[tolinear(boxCoords, isize.x)];
        }
    }   
    //Spostiamo la divisione per eseguirla una volta sola
    outputImage[tolinear(make_int2(pxCoords.x, pxCoords.y), osize.x)] = result * fweight;       
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
    //La prima immagine è l'originale
    std::vector<float4*> dMipChainImages(containers.size());
 
    //Allocazione della memoria necessaria e copia dell'immagine originale
    for(int i = 0; i < containers.size(); ++i){
        cudaMalloc((void **) &(dMipChainImages[i]), containers[i].raw_data_length());
    }
    cudaMemcpy((void *) dMipChainImages[0], (const void *) original.raw_data(), original.raw_data_length(), cudaMemcpyHostToDevice);

    const int block_size = 32;
    for(int i = 0; i < dMipChainImages.size() - 1; ++i){
        //calcolo del numero di blocchi
        dim3 block_dims(block_size, block_size, 1);
        float block_dimX = ceil(float(containers[i + 1].width()) / block_size);
        float block_dimY = ceil(float(containers[i + 1].height()) / block_size);
        dim3 block_n((unsigned int) block_dimX, (unsigned int) block_dimY, 1);

        int2 isize = make_int2(containers[i].width(), containers[i].height());
        int2 osize = make_int2(containers[i + 1].width(), containers[i + 1].height());
        
        GenerateMipMap<<<block_n, block_dims, 0>>>(dMipChainImages[i], dMipChainImages[i + 1], isize, osize); //Il kernel come punto di sincronizzazione tra blocchi 
    }

    //Copia delle immagini dal device all'host
    for(int i = 0; i < dMipChainImages.size() - 1; i++){
        cudaMemcpy((void *) containers[i+1].raw_data(), (const void *) dMipChainImages[i+1], containers[i+1].raw_data_length(), cudaMemcpyDeviceToHost);    
    }

    end = std::chrono::steady_clock::now();
    measurements.image_processing = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    measurements.image_processing /= 1000;

    start = std::chrono::steady_clock::now();
    for(int i = 1; i < containers.size(); ++i){
        std::string save_dest = destination + "mip_" + std::to_string(containers[i].width()) + "x" + std::to_string(containers[i].height()) + "_" + filename;
        ImageUtils::save(containers[i], save_dest.c_str());
    }
    end = std::chrono::steady_clock::now();
    measurements.image_writing = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count(); 
    measurements.image_writing /= 1000;

    std::for_each(dMipChainImages.begin(), dMipChainImages.end(), [](float4* image){ cudaFree(image);});
}

int main(int argc, char** argv){
    
    const std::string filepath = std::string(argv[1]) + std::string(argv[3]);
    
    auto img = ImageUtils::load(filepath.c_str());

    std::cout << "algorithm" << "," << "image" << "," << "width" << "," << "height" << "," << "reading" << "," << "processing" << "," << "writing" << "\n";
    int runs = 10;
    for(int run = 1; run <= runs; run++){
        times times;
        generateMipMapChain(std::string(argv[1]), std::string(argv[2]), std::string(argv[3]), times);        
        cudaDeviceSynchronize();
        std::cout 
        << "CUDA" << ", " 
        << argv[3] << " , " 
        << img.width() << " , " << img.height() << " , " 
        << times.image_reading << ", " 
        << times.image_processing << ", "
        << times.image_writing << ""<< "\n";    }
    checkCudaError();
    return 0;
}