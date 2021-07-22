#include <iostream>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <omp.h>
#include "libs/times.cpp"
#include "libs/ImageUtils.cu"
#include "libs/operators.cu"
#include "libs/mipchain.cu"

void checkCudaError(){
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("GPUassert: %s\n", cudaGetErrorString(err));
    }
}

__global__ void GenerateMipMap(const float4 *inputImage, float4 *outputImage, const int2 isize, const int2 osize, const int2 ioffset, const int2 ooffset){
    const int fwidth = 2;
    const int fsize = (fwidth * fwidth);
    const float fweight = 1.0 / (float) fsize;

    const int2 blockStartCoords = make_int2(blockIdx.x * blockDim.x, blockIdx.y * blockDim.y);
    const int2 pxCoords = make_int2(blockStartCoords.x + threadIdx.x, blockStartCoords.y + threadIdx.y);
    
    if(pxCoords.x >= osize.x || pxCoords.y >= osize.y) return;

    float4 result = make_float4(0, 0, 0, 0);
    for(int row = 0; row < fwidth; ++row){
        for(int column = 0; column < fwidth; ++column){
            int2 boxOffset = make_int2(fwidth * pxCoords.x, fwidth * pxCoords.y);
            int2 boxCoords = make_int2(boxOffset.x + row, boxOffset.y + column);

            boxCoords.x = max(boxCoords.x, 0);
            boxCoords.y = max(boxCoords.y, 0);
            boxCoords.x = min(boxCoords.x, isize.x - 1);
            boxCoords.y = min(boxCoords.y, isize.y - 1);

            //Teniamo conto dell'offset nell'immagine di input
            result = result + inputImage[tolinear(ioffset + boxCoords, isize.x)];
        }
    }

    //Offset nell'immagine di output
    outputImage[tolinear(ooffset + pxCoords, osize.x)] = result * fweight;
       
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

    const int block_size = 32;
    const int n_bands = 4;  

    //Uno stream per ogni banda dell'immagine
    std::vector<cudaStream_t> streams(n_bands);
    std::for_each(streams.begin(), streams.end(), [](cudaStream_t& stream){cudaStreamCreate(&stream);});

    //Un evento per ogni banda di ogni immagine
    //La prima immagine avrà quindi n eventi associati [banda_1, ... , banda_n]
    std::vector<cudaEvent_t> band_work_finished(n_bands * containers.size());
    std::for_each(band_work_finished.begin(), band_work_finished.end(), [](cudaEvent_t& event){cudaEventCreate(&event);});

    //Prima immagine è l'originale
    std::vector<float4*> dMipChainImages(containers.size());
    for(int i = 0; i < containers.size(); ++i){
        cudaMalloc((void **) &(dMipChainImages[i]), containers[i].raw_data_length());
    }

    //Registriamo parallelamente i comandi di trasferimento delle bande
    //sugli stream
    #pragma omp parallel for
    for(int band = 0; band < n_bands; band++){
        //Offset e lunghezza delle bande nella memoria contigua dell'immagine
        unsigned int band_length_bytes = containers[0].width() * (containers[0].height() / n_bands) * containers[0].channels() * sizeof(float);
        unsigned int band_offset_bytes = band_length_bytes * band;
        if((band_offset_bytes + band_length_bytes) > containers[0].raw_data_length()){
            band_length_bytes = containers[0].raw_data_length() - band_offset_bytes;
        }
        char* container_pointer = (char *) containers[0].raw_data();
        container_pointer += band_offset_bytes;
        const char* device_data_pointer = (char *) dMipChainImages[0];
        device_data_pointer += band_offset_bytes;

        //Le bande sono orizzontali così possiamo effettuare una copia di memoria contigua
        cudaMemcpyAsync((void *) device_data_pointer, (const void *) container_pointer, band_length_bytes, cudaMemcpyHostToDevice, streams[band]);
    }

    //Definiamo un limite di dimensioni sotto il quale
    //le immagini non verranno più spezzate in bande
    int2 image_size_threeshold = make_int2(256, 256);
    int n_mips_above_threeshold = std::count_if(containers.begin(), containers.end(), [&](const Image& i){ return i.width() > image_size_threeshold.x && i.height() > image_size_threeshold.y;});

    for(int i = 0; i < containers.size() - 1; ++i){
        const dim3 block_dims(block_size, block_size, 1);
    
        if(i < n_mips_above_threeshold){
            //Dividiamo l'immagine in bande 
            #pragma omp parallel for schedule(static) num_threads(n_bands)
            for(int band = 0; band < n_bands; band++){

                //Calcolo degli offset delle bande sia in input sia in output
                int2 band_ioffset = make_int2(0, (containers[i].height() / n_bands) * band);
                int2 band_ooffset = make_int2(0, (containers[i + 1].height() / n_bands) * band);
                int2 band_isize = make_int2(containers[i].width(), containers[i].height() / n_bands);
                int2 band_osize = make_int2(containers[i + 1].width(), containers[i + 1].height() / n_bands);

                //Blocchi per processare una banda
                float band_block_dimX = ceil(band_isize.x / block_size);
                float band_block_dimY = ceil(band_isize.y / block_size);
                dim3 block_n((unsigned int) band_block_dimX, (unsigned int) band_block_dimY, 1);

                float block_dimX = ceil(float(containers[i + 1].width()) / block_size);
                float block_dimY = ceil(float(containers[i + 1].height() / n_bands) / block_size);

                unsigned int band_length_bytes = containers[i + 1].width() * containers[i + 1].channels() * (containers[i + 1].height() / n_bands) * sizeof(float);
                unsigned int band_offset_bytes = band_length_bytes * band;
                if((band_offset_bytes + band_length_bytes) > containers[i + 1].raw_data_length()){
                    band_length_bytes = containers[i + 1].raw_data_length() - band_offset_bytes;
                }

                //Cast a char* per eseguire i calcoli in byte
                char* container_pointer = (char *) containers[i + 1].raw_data(); 
                container_pointer += band_offset_bytes;
                const char* device_data_pointer = (char *) dMipChainImages[i + 1];
                device_data_pointer += band_offset_bytes;

                if(i > 0){
                    //Se l'immagine non è la prima aspettiamo che tutte le bande finiscano
                    //Ogni stream aspetterà la fine dell'elaborazione delle precedenti n_bands divise su più stream
                    for(int k = 0; k < n_bands; k++){
                        cudaStreamWaitEvent(streams[band], band_work_finished[n_bands * (i - 1) + k]);
                    }
                }
                GenerateMipMap<<<block_n, block_dims, 0, streams[band]>>>(dMipChainImages[i], dMipChainImages[i + 1], band_isize, band_osize, band_ioffset, band_ooffset);
                //Segnaliamo la fine del lavoro su questa banda
                cudaEventRecord(band_work_finished[n_bands * i + band], streams[band]); 
                //La copia deve aspettare che la banda finisca di essere processata, non serve sincronizzazione dato che inviamo i comandi sullo stesso stream
                cudaMemcpyAsync((void *) container_pointer, (const void *) device_data_pointer, band_length_bytes, cudaMemcpyDeviceToHost, streams[band]);
            }
        }else{
            //Immagine troppo piccola, dividerla in bande causerebbe troppo overhead

            int2 ioffset = make_int2(0, 0);
            int2 ooffset = make_int2(0, 0);
            int2 isize = make_int2(containers[i].width(), containers[i].height());
            int2 osize = make_int2(containers[i + 1].width(), containers[i + 1].height());
    
            float block_dimX = ceil(float(containers[i + 1].width()) / block_size);
            float block_dimY = ceil(float(containers[i + 1].height()) / block_size);
            dim3 block_n((unsigned int) block_dimX, (unsigned int) block_dimY, 1);
    
            //Se è la prima immagine che non viene divisa in bande 
            //dobbiamo aspettare che tutte le bande precedenti 
            //abbiano finito
            if(i == n_mips_above_threeshold){
                for(int k = i; k < n_bands; k++){
                    cudaStreamWaitEvent(streams[0], band_work_finished[n_bands * (i - 1) + k]);
                }
            }
            //Eseguiamo il kernel sullo stream 0, non serve sincronizzare questi lanci del kernel GenerateMipMap dato che sono tutti sullo stesso stream
            GenerateMipMap<<<block_n, block_dims, 0, streams[0]>>>(dMipChainImages[i], dMipChainImages[i + 1], isize, osize, ioffset, ooffset); //Kernel as a global sync point
            //E segnaliamo che il lavoro è finito tramite un evento
            cudaEventRecord(band_work_finished[n_bands * i + 0], streams[0]);
            //La copia di questa immagine deve aspettare che il lavoro sia compiuto
            cudaStreamWaitEvent(streams[1], band_work_finished[n_bands * i + 0]);
            //Quando il lavoro è compiuto trasferiamo l'immagine attraverso lo stream 1
            cudaMemcpyAsync((void *) containers[i + 1].raw_data(), (const void *) dMipChainImages[i + 1], containers[i + 1].raw_data_length(), cudaMemcpyDeviceToHost, streams[1]);
        }

    }

    //Aspettiamo che tutte le copie siano state effettuate
    for(cudaStream_t& stream : streams){
        cudaStreamSynchronize(stream);
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

    std::for_each(dMipChainImages.begin(), dMipChainImages.end(), [](float4* image){ cudaFree(image);});

    std::for_each(streams.begin(), streams.end(), [](cudaStream_t& stream){cudaStreamDestroy(stream);});
    std::for_each(band_work_finished.begin(), band_work_finished.end(), [](cudaEvent_t& event){cudaEventDestroy(event);});
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
        << "OpenMP + CUDA async" << ", " 
        << argv[3] << " , " 
        << img.width() << " , " << img.height() << " , " 
        << times.image_reading << ", " 
        << times.image_processing << ", "
        << times.image_writing << ""<< "\n";    }
    checkCudaError();
    return 0;
}