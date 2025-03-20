#include "benchmark.hpp"
#include <chrono>
#include <stdio.h>

CUDNNBenchmark::CUDNNBenchmark(
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
    model_ = new Network(cudnn_, batchSize_, numClasses_, inputWidth_, inputHeight_, inChannels_);
    model_->addConvLayer(convOutChannels_, convKernelSize_, convStride_, convPadding_);
    model_->addFCLayer(model_->getFlattenedSize(), numClasses_);
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