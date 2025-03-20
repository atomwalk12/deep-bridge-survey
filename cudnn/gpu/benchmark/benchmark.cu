#include "benchmark.hpp"
#include <chrono>
#include <stdio.h>


CUDNNBenchmark::CUDNNBenchmark(
    std::string config_path,
    int batchSize,
    int numClasses,
    int inChannels,
    int inputHeight,
    int inputWidth,
    int convOutChannels,
    int convKernelSize,
    int convStride,
    int convPadding,
    int warmupIterations,
    int benchmarkIterations
) : 
    config_path_(config_path),
    batchSize_(batchSize),
    numClasses_(numClasses),
    inChannels_(inChannels),
    inputHeight_(inputHeight),
    inputWidth_(inputWidth),
    warmupIterations_(warmupIterations),
    benchmarkIterations_(benchmarkIterations),
    inputSize_(batchSize * inChannels * inputWidth * inputHeight),
    outputSize_(batchSize * numClasses),
    inputGradientSize_(batchSize * inChannels * inputWidth * inputHeight)
{
    checkCUDNN(cudnnCreate(&cudnn_));
    
    // Load configuration first
    std::map<std::string, int> params;
    std::vector<std::vector<int>> convLayers;
    std::vector<std::vector<int>> fcLayers;
    
    if (!loadSimpleConfig(config_path_, params, convLayers, fcLayers)) {
        throw std::runtime_error("Failed to load configuration file");
    }

    // Update parameters from config file if they exist
    if (params.count("BATCH_SIZE")) batchSize_ = params["BATCH_SIZE"];
    if (params.count("NUM_CLASSES")) numClasses_ = params["NUM_CLASSES"];
    if (params.count("IN_CHANNELS")) inChannels_ = params["IN_CHANNELS"];
    if (params.count("INPUT_HEIGHT")) inputHeight_ = params["INPUT_HEIGHT"];
    if (params.count("INPUT_WIDTH")) inputWidth_ = params["INPUT_WIDTH"];
    if (params.count("WARMUP_ITERATIONS")) warmupIterations_ = params["WARMUP_ITERATIONS"];
    if (params.count("NUM_ITERATIONS")) benchmarkIterations_ = params["NUM_ITERATIONS"];

    // Update derived parameters
    inputSize_ = batchSize_ * inChannels_ * inputWidth_ * inputHeight_;
    outputSize_ = batchSize_ * numClasses_;
    inputGradientSize_ = batchSize_ * inChannels_ * inputWidth_ * inputHeight_;

    initializeNetwork(convLayers, fcLayers);
    allocateMemory();
    initializeData();
}

void CUDNNBenchmark::initializeNetwork(
    const std::vector<std::vector<int>>& convLayers,
    const std::vector<std::vector<int>>& fcLayers
) {
    model_ = new Network(cudnn_, batchSize_, numClasses_, inputWidth_, inputHeight_, inChannels_);
    
    for (const auto& layer : convLayers) {
        model_->addConvLayer(layer[0], layer[1], layer[2], layer[3]);
    }
    
    // Add FC layers
    int prev_size = model_->getFlattenedSize();
    for (const auto& layer : fcLayers) {
        int out_features = layer[0];
        if (out_features == -1) {
            out_features = numClasses_;
        }
        model_->addFCLayer(prev_size, out_features);
        prev_size = out_features;
    }
}

void CUDNNBenchmark::allocateMemory() {
    cudaMallocManaged(&inputData_, inputSize_ * sizeof(float));
    cudaMallocManaged(&outputData_, outputSize_ * sizeof(float));
    cudaMallocManaged(&targetData_, outputSize_ * sizeof(float));
    cudaMallocManaged(&inputGradient_, inputGradientSize_ * sizeof(float));
    outputGradient_ = model_->createDummyGradient(outputData_);
}

void CUDNNBenchmark::initializeData() {
    // Initialize input data
    for (int b = 0; b < batchSize_; b++) {
        for (int c = 0; c < inChannels_; c++) {
            for (int h = 0; h < inputHeight_; h++) {
                for (int w = 0; w < inputWidth_; w++) {
                    int idx = ((b * inChannels_ + c) * inputHeight_ + h) * inputWidth_ + w;
                    inputData_[idx] = (float)rand() / RAND_MAX;
                }
            }
        }
    }

    // Initialize output and target data
    for (int i = 0; i < outputSize_; i++) {
        outputData_[i] = 0.0f;
        targetData_[i] = 1.0f;
    }
}

void CUDNNBenchmark::runWarmup() {
    MSELoss loss;
    CostHistory costHistory;
    cost_history_init(&costHistory);

    for (int i = 0; i < warmupIterations_; i++) {
        model_->zeroGradients();
        model_->forward(inputData_, outputData_);
        
        float lossValue = loss.compute(outputData_, targetData_, outputSize_);
        
        if (i % 10 == 0)
            cost_history_add(&costHistory, lossValue);
            
        loss.backward(outputData_, targetData_, outputGradient_, outputSize_);
        
        model_->backwardInput(inputGradient_, outputGradient_);
        model_->backwardParams(inputData_, outputGradient_);
        model_->updateWeights(0.001f);
        
        printf("Iteration %d, Loss: %f\n", i, lossValue);
    }
    
    plot_cost_ascii(&costHistory);
    cudaDeviceSynchronize();
}

double CUDNNBenchmark::timeForwardPass() {
    auto begin = std::chrono::steady_clock::now();
    
    for (int i = 0; i < benchmarkIterations_; i++) {
        model_->forward(inputData_, outputData_);
    }
    cudaDeviceSynchronize();
    
    auto end = std::chrono::steady_clock::now();
    double totalMicroseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    return (totalMicroseconds / 1000.0) / benchmarkIterations_;
}

double CUDNNBenchmark::timeBackwardInputPass() {
    auto begin = std::chrono::steady_clock::now();
    
    for (int i = 0; i < benchmarkIterations_; i++) {
        model_->backwardInput(inputGradient_, outputGradient_);
    }
    cudaDeviceSynchronize();
    
    auto end = std::chrono::steady_clock::now();
    double totalMicroseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    return (totalMicroseconds / 1000.0) / benchmarkIterations_;
}

double CUDNNBenchmark::timeBackwardParamsPass() {
    auto begin = std::chrono::steady_clock::now();
    
    for (int i = 0; i < benchmarkIterations_; i++) {
        model_->backwardParams(inputData_, outputGradient_);
    }
    cudaDeviceSynchronize();
    
    auto end = std::chrono::steady_clock::now();
    double totalMicroseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    return (totalMicroseconds / 1000.0) / benchmarkIterations_;
}

void CUDNNBenchmark::runBenchmark() {
    runWarmup();
    
    double forwardTime = timeForwardPass();
    printf("Average forward pass time: %f ms\n", forwardTime);
    
    double backwardInputTime = timeBackwardInputPass();
    printf("Average backward input pass time: %f ms\n", backwardInputTime);
    
    double backwardParamsTime = timeBackwardParamsPass();
    printf("Average backward params pass time: %f ms\n", backwardParamsTime);
    
    double totalTime = forwardTime + backwardInputTime + backwardParamsTime;
    printf("Total time: %f ms\n", totalTime);
}

void CUDNNBenchmark::cleanup() {
    cudaFree(inputData_);
    cudaFree(outputData_);
    cudaFree(inputGradient_);
    cudaFree(outputGradient_);
    cudaFree(targetData_);
    delete model_;
    cudnnDestroy(cudnn_);
}

CUDNNBenchmark::~CUDNNBenchmark() {
    cleanup();
} 