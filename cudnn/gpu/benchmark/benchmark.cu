#include "benchmark.hpp"
#include <chrono>
#include <stdio.h>

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
    convOutChannels_(convOutChannels),
    convKernelSize_(convKernelSize),
    convStride_(convStride),
    convPadding_(convPadding),
    warmupIterations_(warmupIterations),
    benchmarkIterations_(benchmarkIterations),
    inputSize_(batchSize * inChannels * inputWidth * inputHeight),
    outputSize_(batchSize * numClasses),
    inputGradientSize_(batchSize * inChannels * inputWidth * inputHeight)
{
    checkCUDNN(cudnnCreate(&cudnn_));
    initializeNetwork();
    allocateMemory();
    initializeData();
} 

void CUDNNBenchmark::initializeNetwork() {
    // Load configuration
    std::map<std::string, int> params;
    std::vector<std::vector<int>> convLayers;
    std::vector<std::vector<int>> fcLayers;
    
    if (!loadSimpleConfig(config_path_, params, convLayers, fcLayers)) {
        throw std::runtime_error("Failed to load configuration file");
    }

    model_ = new Network(cudnn_, batchSize_, numClasses_, inputWidth_, inputHeight_, inChannels_);
    for (const auto& layer : convLayers) {
        model_->addConvLayer(layer[0], layer[1], layer[2], layer[3]);
    }
    
    // Add FC layers
    int prev_size = model_->getFlattenedSize();
    for (const auto& layer : fcLayers) {
        int out_features = layer[0];
        if (out_features == -1) {
            out_features = NUM_CLASSES;
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