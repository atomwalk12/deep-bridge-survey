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

const int INPUT_SIZE = BATCH_SIZE * IN_CHANNELS * INPUT_WIDTH * INPUT_HEIGHT;
const int OUTPUT_SIZE = BATCH_SIZE * NUM_CLASSES;
const int INPUT_GRADIENT_SIZE = BATCH_SIZE * IN_CHANNELS * INPUT_WIDTH * INPUT_HEIGHT;


int main(int argc, char **argv)
{
    // Default configuration file path
    std::string config_path = "gpu/network_config.txt"; // Assume the program is running from the cudnn folder
    
    if (argc > 1) {
        config_path = argv[1]; // if command line argument is provided
    }
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
    // Load configuration
    std::map<std::string, int> params;
    std::vector<std::vector<int>> convLayers;
    std::vector<std::vector<int>> fcLayers;
    
    if (!loadSimpleConfig(config_path, params, convLayers, fcLayers)) {
        throw std::runtime_error("Failed to load configuration file");
    }

    Network model(cudnn, BATCH_SIZE, NUM_CLASSES, INPUT_WIDTH, INPUT_HEIGHT, IN_CHANNELS);
    for (const auto& layer : convLayers) {
        model.addConvLayer(layer[0], layer[1], layer[2], layer[3]);
    }
    
    // Add FC layers
    int prev_size = model.getFlattenedSize();
    for (const auto& layer : fcLayers) {
        int out_features = layer[0];
        if (out_features == -1) {
            out_features = NUM_CLASSES;
        }
        model.addFCLayer(prev_size, out_features);
        prev_size = out_features;
    }

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