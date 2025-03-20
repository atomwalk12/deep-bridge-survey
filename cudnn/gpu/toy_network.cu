#include <stdio.h>
#include "core/network.h"
#include <chrono>
#include "loss/loss.h"
#include "core/utils.h"

// Benchmark parameters
const int NUM_ITERATIONS = 300;

// Network parameters
const int BATCH_SIZE = 1;
const int NUM_CLASSES = 3;
const int IN_CHANNELS = 1;
const int INPUT_HEIGHT = 3;
const int INPUT_WIDTH = 3;

const int CONV1_OUT_CHANNELS = 16;
const int CONV1_KERNEL_SIZE = 3;
const int CONV1_STRIDE = 1;
const int CONV1_PADDING = 1;

const int CONV2_OUT_CHANNELS = 32;
const int CONV2_KERNEL_SIZE = 3;
const int CONV2_STRIDE = 1;
const int CONV2_PADDING = 1;

const int CONV3_OUT_CHANNELS = 64;
const int CONV3_KERNEL_SIZE = 3;
const int CONV3_STRIDE = 1;
const int CONV3_PADDING = 1;

const int INPUT_SIZE = BATCH_SIZE * IN_CHANNELS * INPUT_WIDTH * INPUT_HEIGHT;
const int OUTPUT_SIZE = BATCH_SIZE * NUM_CLASSES;
const int INPUT_GRADIENT_SIZE = BATCH_SIZE * IN_CHANNELS * INPUT_WIDTH * INPUT_HEIGHT;


int main()
{

    // ==============================
    // Initialization
    // ==============================
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    CostHistory cost_history;
    cost_history_init(&cost_history);

    // Create input data
    float *input_data, *output_data;

    cudaMallocManaged(&input_data, INPUT_SIZE * sizeof(float));
    cudaMallocManaged(&output_data, OUTPUT_SIZE * sizeof(float));

    float static_input[BATCH_SIZE][IN_CHANNELS][INPUT_HEIGHT][INPUT_WIDTH] = {
        {// batch 0
         {
             // channel 0
             {1.0f, 0.0f, 1.0f}, // row 0
             {0.0f, 1.0f, 0.0f}, // row 1
             {1.0f, 0.0f, 1.0f}  // row 2
         }}};

    for (int i = 0; i < INPUT_SIZE; i++)
    {
        input_data[i] = ((float *)static_input)[i];
    }

    // Create target data
    float *target_data;
    cudaMallocManaged(&target_data, OUTPUT_SIZE * sizeof(float));

    // Use one-hot encoding where 1 represents the correct class
    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        target_data[i] = (i == 0) ? 1.0f : 0.0f;
    }

    // ==============================
    // Create the network
    // ==============================
    Network model(cudnn, BATCH_SIZE, NUM_CLASSES, INPUT_WIDTH, INPUT_HEIGHT, IN_CHANNELS);

    model.addConvLayer(CONV1_OUT_CHANNELS, CONV1_KERNEL_SIZE, CONV1_STRIDE, CONV1_PADDING);
    model.addConvLayer(CONV2_OUT_CHANNELS, CONV2_KERNEL_SIZE, CONV2_STRIDE, CONV2_PADDING);
    model.addConvLayer(CONV3_OUT_CHANNELS, CONV3_KERNEL_SIZE, CONV3_STRIDE, CONV3_PADDING);
    model.addFCLayer(model.getFlattenedSize(), 512);
    model.addFCLayer(512, 128);
    model.addFCLayer(128, NUM_CLASSES);

    float *output_gradient = model.createDummyGradient(output_data);
    float *input_gradient;

    cudaMallocManaged(&input_gradient, INPUT_GRADIENT_SIZE * sizeof(float));
    cudaDeviceSynchronize();

    // ==============================
    // Training
    // ==============================
    MSELoss loss;

    for (int i = 0; i < NUM_ITERATIONS; i++)
    {
        model.zeroGradients();

        model.forward(input_data, output_data);

        float loss_value = loss.compute(output_data, target_data, OUTPUT_SIZE);

        if (i % 10 == 0)
            cost_history_add(&cost_history, loss_value);

        loss.backward(output_data, target_data, output_gradient, OUTPUT_SIZE);

        model.backwardInput(input_gradient, output_gradient);
        model.backwardParams(input_data, output_gradient);

        model.updateWeights(0.001f);

        printf("Iteration %d, Loss: %f\n", i, loss_value);
    }

    plot_cost_ascii(&cost_history);

    cudaDeviceSynchronize();
    cudaFree(target_data);
}