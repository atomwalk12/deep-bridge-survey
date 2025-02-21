#ifndef NETWORK_H
#define NETWORK_H

#include <cudnn.h>
#include <cublas_v2.h>
#include <vector>
#include "layer.h"
#include "conv_layer.h"
#include "fc_layer.h"

class Network {
public:
    Network(cudnnHandle_t& handle, int batch_size, int num_classes, int initial_width, int initial_height, int initial_channels);
    ~Network();
    
    void addConvLayer(int out_channels, int kernel_size, int stride, int padding);
    void addFCLayer(int in_features, int out_features);
    
    void forward(float *inp, float *out);
    float* createDummyGradient(float* output);
    void backwardInput(float *inp_grad, float *out_grad);
    void backwardParams(float *inp, float *out_grad);
    void updateWeights(float learning_rate);
    void zeroGradients();
    
    int getOutputSize() const {
        return batch_size_ * num_classes_;
    }
    
    int getFlattenedSize() const {
        return current_channels_ * current_height_ * current_width_;
    }
    
private:
    cudnnHandle_t& cudnn;
    cublasHandle_t cublas;
    
    std::vector<Layer*> layers;
    std::vector<float*> layer_outputs;    // outputs between layers
    std::vector<float*> gradient_outputs; // gradients between layers

    int batch_size_;
    int num_classes_;
    int current_width_;
    int current_height_;
    int current_channels_;
};

#endif // NETWORK_H 