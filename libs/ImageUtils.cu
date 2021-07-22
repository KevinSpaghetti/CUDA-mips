#pragma once

#include <string>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "libs/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "libs/stb_image_write.h"
#include "Image.cu"

namespace ImageUtils{

    Image load(const char* filepath){
        int width;
        int height;
        int channels;

        stbi_ldr_to_hdr_gamma(1.0f);
        float* image = stbi_loadf(filepath, &width, &height, &channels, STBI_rgb_alpha);
        if(stbi_failure_reason()){
            std::cout << stbi_failure_reason();
            exit(1);
        }

        Image result(width, height, 4, image);

        stbi_image_free(image);

        return result;
    }

    void save(Image& image, const char* destination_filepath){
        const int image_size = image.width() * image.height() * image.channels() * sizeof(unsigned char);
        unsigned char* converted_data = (unsigned char*) malloc(image_size);
        float *raw_data = image.raw_data();
        int i = 0;
        int j = 0;
        while(i < image_size - (image_size / 4)){
            converted_data[i] = (unsigned char) int(raw_data[j] * 255.0); //r
            ++i;
            ++j;
            converted_data[i] = (unsigned char) int(raw_data[j] * 255.0); //g
            ++i;
            ++j;
            converted_data[i] = (unsigned char) int(raw_data[j] * 255.0); //b
            ++i;
            ++j;
            ++j; //Skip alpha
        }
        stbi_write_jpg(destination_filepath,
            image.width(), image.height(), 
            STBI_rgb, 
            converted_data, 100);
        free(converted_data);
        if(stbi_failure_reason()){
            std::cout << stbi_failure_reason();
            exit(1);
        }    
    }

};

