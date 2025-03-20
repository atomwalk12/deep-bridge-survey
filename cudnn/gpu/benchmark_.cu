#include <stdio.h>
#include "core/network.h"
#include <chrono>
#include "loss/loss.h"
#include "core/utils.h"

// Benchmark parameters
int NUM_ITERATIONS = 300;
int WARMUP_ITERATIONS = 300;

// Network parameters
int BATCH_SIZE = 1;
int NUM_CLASSES = 3;
int IN_CHANNELS = 1;
int INPUT_HEIGHT = 3;
int INPUT_WIDTH = 3;

int CONV1_OUT_CHANNELS = 16;
int CONV1_KERNEL_SIZE = 3;
int CONV1_STRIDE = 1;
int CONV1_PADDING = 1;

int CONV2_OUT_CHANNELS = 32;
int CONV2_KERNEL_SIZE = 3;
int CONV2_STRIDE = 1;
int CONV2_PADDING = 1;

int CONV3_OUT_CHANNELS = 64;
int CONV3_KERNEL_SIZE = 3;
int CONV3_STRIDE = 1;
int CONV3_PADDING = 1;

int INPUT_SIZE = BATCH_SIZE * IN_CHANNELS * INPUT_WIDTH * INPUT_HEIGHT;
int OUTPUT_SIZE = BATCH_SIZE * NUM_CLASSES;
int INPUT_GRADIENT_SIZE = BATCH_SIZE * IN_CHANNELS * INPUT_WIDTH * INPUT_HEIGHT;


int main()
{
    std::chrono::steady_clock::time_point begin, end;

    // Initialize CUDNN
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    // Create network
    Network model = buildNetworkFromConfig("network_config.txt", cudnn, NUM_ITERATIONS, WARMUP_ITERATIONS, BATCH_SIZE, NUM_CLASSES, INPUT_SIZE, OUTPUT_SIZE, INPUT_GRADIENT_SIZE);
    Network model(cudnn, BATCH_SIZE, NUM_CLASSES, INPUT_WIDTH, INPUT_HEIGHT, IN_CHANNELS);

    model.addConvLayer(CONV1_OUT_CHANNELS, CONV1_KERNEL_SIZE, CONV1_STRIDE, CONV1_PADDING);
    model.addConvLayer(CONV2_OUT_CHANNELS, CONV2_KERNEL_SIZE, CONV2_STRIDE, CONV2_PADDING);
    model.addConvLayer(CONV3_OUT_CHANNELS, CONV3_KERNEL_SIZE, CONV3_STRIDE, CONV3_PADDING);
    model.addFCLayer(model.getFlattenedSize(), 512);
    model.addFCLayer(512, 128);
    model.addFCLayer(128, NUM_CLASSES);

    // Initialize cost history
    CostHistory cost_history;
    cost_history_init(&cost_history);

    // ================================
    // Input data
    // ================================
    float *input_data, *output_data;

    cudaMallocManaged(&input_data, INPUT_SIZE * sizeof(float));
    cudaMallocManaged(&output_data, OUTPUT_SIZE * sizeof(float));

    // Create input data with explicit layout
    for (int b = 0; b < BATCH_SIZE; b++)
    {
        for (int c = 0; c < IN_CHANNELS; c++)
        {
            for (int h = 0; h < INPUT_HEIGHT; h++)
            {
                for (int w = 0; w < INPUT_WIDTH; w++)
                {
                    int idx = ((b * IN_CHANNELS + c) * INPUT_HEIGHT + h) * INPUT_WIDTH + w;
                    input_data[idx] = (float)rand() / RAND_MAX;
                }
            }
        }
    }

    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        output_data[i] = 0.0f;
    }

    // Create dummy gradient for backward pass
    float *output_gradient = model.createDummyGradient(output_data);
    float *input_gradient;
    cudaMallocManaged(&input_gradient, INPUT_GRADIENT_SIZE * sizeof(float));
    cudaDeviceSynchronize();

    MSELoss loss;

    // ================================
    // Warmup run
    // ================================

    // Target data still needs size, but we get it from the model
    float *target_data;
    cudaMallocManaged(&target_data, OUTPUT_SIZE * sizeof(float));

    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        target_data[i] = 1.0f;
    }

    for (int i = 0; i < WARMUP_ITERATIONS; i++)
    {
        model.zeroGradients();

        // Forward pass
        model.forward(input_data, output_data);

        // Compute loss and gradients
        float loss_value = loss.compute(output_data, target_data, OUTPUT_SIZE);

        if (i % 10 == 0)
            cost_history_add(&cost_history, loss_value);

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

    // ==================================
    // Forward pass
    // ==================================
    begin = std::chrono::steady_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; i++)
    {
        model.forward(input_data, output_data);
    }
    cudaDeviceSynchronize();
    end = std::chrono::steady_clock::now();

    double total_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    double average_milliseconds = (total_microseconds / 1000.0) / NUM_ITERATIONS;
    double total_time = average_milliseconds;
    printf("Average forward pass time: %f ms\n", average_milliseconds);

    // ================================
    // Backward input pass
    // ================================
    begin = std::chrono::steady_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; i++)
    {
        model.backwardInput(input_gradient, output_gradient);
    }
    cudaDeviceSynchronize();
    end = std::chrono::steady_clock::now();

    total_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    average_milliseconds = (total_microseconds / 1000.0) / NUM_ITERATIONS;
    total_time += average_milliseconds;
    printf("Average backward input pass time: %f ms\n", average_milliseconds);

    // ================================
    // Backward params pass
    // ================================
    begin = std::chrono::steady_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; i++)
    {
        model.backwardParams(input_data, output_gradient);
    }
    cudaDeviceSynchronize();
    end = std::chrono::steady_clock::now();

    total_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    average_milliseconds = (total_microseconds / 1000.0) / NUM_ITERATIONS;
    total_time += average_milliseconds;
    printf("Average backward params pass time: %f ms\n", average_milliseconds);
    printf("Total time: %f ms\n", total_time);

    // Additional cleanup
    cudaFree(input_gradient);
    cudaFree(output_gradient);

    // Cleanup
    cudaFree(input_data);
    cudaFree(output_data);
    cudnnDestroy(cudnn);

    return 0;
}
