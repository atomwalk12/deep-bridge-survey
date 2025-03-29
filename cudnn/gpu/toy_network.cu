#include <stdio.h>
#include "core/network.h"
#include <chrono>
#include "loss/loss.h"
#include "core/utils.h"

int main(int argc, char **argv)
{
    // Default configuration file path
    std::string config_path = "gpu/network_config.txt"; // Assume the program is running from the cudnn folder
    
    if (argc > 1) {
        config_path = argv[1]; // if command line argument is provided
    }
    // Load configuration
    std::map<std::string, int> params;
    std::vector<std::vector<int>> convLayers;
    std::vector<std::vector<int>> fcLayers;
    
    if (!loadSimpleConfig(config_path, params, convLayers, fcLayers)) {
        throw std::runtime_error("Failed to load configuration file");
    }


    int batchSize_, numClasses_, inChannels_, inputHeight_, inputWidth_;
    int benchmarkIterations_, inputSize_, outputSize_, inputGradientSize_;
    // Update parameters from config file if they exist
    if (params.count("BATCH_SIZE")) batchSize_ = params["BATCH_SIZE"];
    if (params.count("NUM_CLASSES")) numClasses_ = params["NUM_CLASSES"];
    if (params.count("IN_CHANNELS")) inChannels_ = params["IN_CHANNELS"];
    if (params.count("INPUT_HEIGHT")) inputHeight_ = params["INPUT_HEIGHT"];
    if (params.count("INPUT_WIDTH")) inputWidth_ = params["INPUT_WIDTH"];
    if (params.count("NUM_ITERATIONS")) benchmarkIterations_ = params["NUM_ITERATIONS"];
    
    // Update derived parameters
    inputSize_ = batchSize_ * inChannels_ * inputWidth_ * inputHeight_;
    outputSize_ = batchSize_ * numClasses_;
    inputGradientSize_ = batchSize_ * inChannels_ * inputWidth_ * inputHeight_;


    // ==============================
    // Initialization
    // ==============================
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    CostHistory cost_history;
    cost_history_init(&cost_history);

    // Create input data
    float *input_data, *output_data;

    cudaMallocManaged(&input_data, inputSize_ * sizeof(float));
    cudaMallocManaged(&output_data, outputSize_ * sizeof(float));

    float static_input[batchSize_][inChannels_][inputHeight_][inputWidth_] = {
        {// batch 0
         {
             // channel 0
             {1.0f, 0.0f, 1.0f}, // row 0
             {0.0f, 1.0f, 0.0f}, // row 1
             {1.0f, 0.0f, 1.0f}  // row 2
         }}};

    for (int i = 0; i < inputSize_; i++)
    {
        input_data[i] = ((float *)static_input)[i];
    }

    // Create target data
    float *target_data;
    cudaMallocManaged(&target_data, outputSize_ * sizeof(float));

    // Use one-hot encoding where 1 represents the correct class
    for (int i = 0; i < outputSize_; i++)
    {
        target_data[i] = (i == 0) ? 1.0f : 0.0f;
    }

    // ==============================
    // Create the network
    // ==============================
    Network model(cudnn, batchSize_, numClasses_, inputWidth_, inputHeight_, inChannels_);
    for (const auto& layer : convLayers) {
        model.addConvLayer(layer[0], layer[1], layer[2], layer[3]);
    }
    
    // Add FC layers
    int prev_size = model.getFlattenedSize();
    for (const auto& layer : fcLayers) {
        int out_features = layer[0];
        if (out_features == -1) {
            out_features = numClasses_;
        }
        model.addFCLayer(prev_size, out_features);
        prev_size = out_features;
    }

    float *output_gradient = model.createDummyGradient(output_data);
    float *input_gradient;

    cudaMallocManaged(&input_gradient, inputGradientSize_ * sizeof(float));
    cudaDeviceSynchronize();

    // ==============================
    // Training
    // ==============================
    MSELoss loss;

    for (int i = 0; i < benchmarkIterations_; i++)
    {
        model.zeroGradients();

        model.forward(input_data, output_data);

        float loss_value = loss.compute(output_data, target_data, outputSize_);

        if (i % 10 == 0)
            cost_history_add(&cost_history, loss_value);

        loss.backward(output_data, target_data, output_gradient, outputSize_);

        model.backwardInput(input_gradient, output_gradient);
        model.backwardParams(input_data, output_gradient);

        model.updateWeights(0.001f);

        printf("Iteration %d, Loss: %f\n", i, loss_value);
    }

    plot_cost_ascii(&cost_history);

    cudaDeviceSynchronize();
    cudaFree(target_data);
}
