#include <stdio.h>
#include "alexnet.h"
#include <chrono>
#include "loss.h"
#include "utils.h"
// Benchmark parameters
const int NUM_ITERATIONS = 100;
const int WARMUP_ITERATIONS = 300;

// Network parameters
const int BATCH_SIZE = 1;
const int NUM_CLASSES = 3;
const int IN_CHANNELS = 1;
const int INPUT_HEIGHT = 3;
const int INPUT_WIDTH = 3;

// First Conv2D layer parameters
const int CONV1_OUT_CHANNELS = 16;  // Increased from 1
const int CONV1_KERNEL_SIZE = 3;    // Increased from 2
const int CONV1_STRIDE = 1;
const int CONV1_PADDING = 1;

// Second Conv2D layer parameters
const int CONV2_OUT_CHANNELS = 32;
const int CONV2_KERNEL_SIZE = 3;    // Increased from 1
const int CONV2_STRIDE = 1;
const int CONV2_PADDING = 1;

// Third Conv2D layer parameters
const int CONV3_OUT_CHANNELS = 64;
const int CONV3_KERNEL_SIZE = 3;    // Increased from 1
const int CONV3_STRIDE = 1;
const int CONV3_PADDING = 1;

// Define sizes in terms of number of elements
const int INPUT_SIZE = BATCH_SIZE * IN_CHANNELS * INPUT_WIDTH * INPUT_HEIGHT;
const int OUTPUT_SIZE = BATCH_SIZE * NUM_CLASSES;
const int INPUT_GRADIENT_SIZE = BATCH_SIZE * IN_CHANNELS * INPUT_WIDTH * INPUT_HEIGHT;

void checkCUDNN(cudnnStatus_t status) {
    if (status != CUDNN_STATUS_SUCCESS) {
        printf("cuDNN Error: %s\n", cudnnGetErrorString(status));
        exit(1);
    }
}

int main() {
    std::chrono::steady_clock::time_point begin, end;

    // Initialize CUDNN
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    // Create network with initial dimensions
    Network model(cudnn, BATCH_SIZE, NUM_CLASSES, INPUT_WIDTH, INPUT_HEIGHT, IN_CHANNELS);
    
    // Add layers
    model.addConvLayer(CONV1_OUT_CHANNELS, CONV1_KERNEL_SIZE, CONV1_STRIDE, CONV1_PADDING);
    model.addConvLayer(CONV2_OUT_CHANNELS, CONV2_KERNEL_SIZE, CONV2_STRIDE, CONV2_PADDING);
    model.addConvLayer(CONV3_OUT_CHANNELS, CONV3_KERNEL_SIZE, CONV3_STRIDE, CONV3_PADDING);
    
    // Dense layers with gradual size reduction
    model.addFCLayer(model.getFlattenedSize(), 512);
    model.addFCLayer(512, 128);
    model.addFCLayer(128, NUM_CLASSES);

    // Initialize cost history
    CostHistory cost_history;
    cost_history_init(&cost_history);

    // These vectors will be initialized with random values
    // ================================
    // =====      Input data      =====
    // ================================
    float *input_data, *output_data;
    
    cudaMallocManaged(&input_data, INPUT_SIZE * sizeof(float));
    cudaMallocManaged(&output_data, OUTPUT_SIZE * sizeof(float));

    // Create input data with a simple pattern
    float static_input[BATCH_SIZE][IN_CHANNELS][INPUT_HEIGHT][INPUT_WIDTH] = {
        {   // batch 0
            {   // channel 0
                {1.0f, 0.0f, 1.0f},  // row 0
                {0.0f, 1.0f, 0.0f},  // row 1
                {1.0f, 0.0f, 1.0f}   // row 2
            }
        }
    };

    // Copy the static values to input_data
    for (int i = 0; i < INPUT_SIZE; i++) {
        input_data[i] = ((float*)static_input)[i];
    }

    // Create target data with a clear pattern
    // Let's say we want the network to classify this as class 0
    float* target_data;
    cudaMallocManaged(&target_data, OUTPUT_SIZE * sizeof(float));

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        target_data[i] = (i == 0) ? 1.0f : 0.0f;  // One-hot encoding for class 0
    }

    // Initialize output_data to zeros
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output_data[i] = 0.0f;
    }

    // Create dummy gradient for backward pass
    float* output_gradient = model.createDummyGradient(output_data);
    float* input_gradient;
    cudaMallocManaged(&input_gradient, INPUT_GRADIENT_SIZE * sizeof(float));
    cudaDeviceSynchronize();

    MSELoss loss;

    // ================================
    // =====      Warmup run      =====
    // ================================
    // MSELoss loss;
    
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        model.zeroGradients();
        
        // Forward pass
        model.forward(input_data, output_data);
        
        // Compute loss and gradients
        float loss_value = loss.compute(output_data, target_data, OUTPUT_SIZE);

        if (i % 10 == 0) cost_history_add(&cost_history, loss_value);

        loss.backward(output_data, target_data, output_gradient, OUTPUT_SIZE);
        
        // Backward pass
        model.backwardInput(input_gradient, output_gradient);
        model.backwardParams(input_data, output_gradient);
        
        // Update weights
        model.updateWeights(0.001f);
        
        
        printf("Iteration %d, Loss: %f\n", i, loss_value);
    }

    plot_cost_ascii(&cost_history);

    cudaDeviceSynchronize();

    // Cleanup
    cudaFree(target_data);
}