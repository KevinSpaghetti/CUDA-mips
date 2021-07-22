#pragma once

#include <vector>
#include <iostream>

class Image {

private:
    unsigned int _width;
    unsigned int _height;

    unsigned int _channels;
    float* _data;

    bool _usesPinnedMemory;

    Image() = delete;

public:
    Image(const unsigned int width, const unsigned int height, unsigned int channels) :
        _width(width), _height(height), _channels(channels) {
        _usesPinnedMemory = true;
        auto status = cudaMallocHost((void**) &_data, raw_data_length());
        if(status != cudaSuccess){
            std::cerr << "cudaMallocHost failed, pageable memory used" << "\n";
            _usesPinnedMemory = false;
            _data = (float*) malloc(raw_data_length());
        }
    }

    Image(const unsigned int width, const unsigned int height,  unsigned int channels, float* image_data) :
        _width(width), _height(height), _channels(channels) {
        _usesPinnedMemory = true;
        auto status = cudaMallocHost((void**) &_data, raw_data_length());
        if(status != cudaSuccess){
            std::cerr << "cudaMallocHost failed, pageable memory used" << "\n";
            _usesPinnedMemory = false;
            _data = (float*) malloc(raw_data_length());
        }
        memcpy(_data, image_data, raw_data_length());
    }

    unsigned int width() const { return _width; }
    unsigned int height() const { return _height; }
    unsigned int channels() const { return _channels; }

    unsigned int raw_data_length() const { return _width * _height * _channels * sizeof(float); }
    float* raw_data() { return _data; }
    const float* raw_data() const { return _data; }

    Image(const Image &rhs){
        _width = rhs.width();
        _height = rhs.height();
        _channels = rhs.channels();
        if(rhs._usesPinnedMemory){
            _usesPinnedMemory = true;
            auto status = cudaMallocHost((void**) &_data, rhs.raw_data_length());
            if(status != cudaSuccess){
                std::cerr << "cudaMallocHost failed, pegeable memory used" << "\n";
                _usesPinnedMemory = false;
                _data = (float*) malloc(rhs.raw_data_length());
            }
        }
        memcpy((void *) _data, (const void *) rhs.raw_data(), rhs.raw_data_length());
    };
    Image(Image &rhs){
        _width = rhs.width();
        _height = rhs.height();
        _channels = rhs.channels();
        _usesPinnedMemory = rhs._usesPinnedMemory;
        _data = rhs._data;
        rhs._data = nullptr;
    };
    

    ~Image(){
        if(_data != nullptr){
            if(_usesPinnedMemory == true){
                cudaFreeHost(_data);
            }else{
                free(_data);
            }
        }
    }
};