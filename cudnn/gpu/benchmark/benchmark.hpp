#pragma once

#include <cudnn.h>
#include "../core/network.h"
#include "../loss/loss.h"
#include "../core/utils.h"

class CUDNNBenchmark {
public:
    CUDNNBenchmark(
        int batchSize = 1,
        int numClasses = 3,
        int inChannels = 1,
        int inputHeight = 2,
        int inputWidth = 3,
        int convOutChannels = 2,
        int convKernelSize = 2,
        int convStride = 1,
        int convPadding = 0,
        int warmupIterations = 300,
        int benchmarkIterations = 100
    );
    ~CUDNNBenchmark();
    
    void runBenchmark();

private:
    // Network parameters
    const int batchSize_;
    const int numClasses_;
    const int inChannels_;
    const int inputHeight_;
    const int inputWidth_;
    const int convOutChannels_;
    const int convKernelSize_;
    const int convStride_;
    const int convPadding_;
    const int warmupIterations_;
    const int benchmarkIterations_;

    // Derived sizes
    const int inputSize_;
    const int outputSize_;
    const int inputGradientSize_;

    // CUDNN handle
    cudnnHandle_t cudnn_;
    Network* model_;
    
    // Data pointers
    float *inputData_;
    float *outputData_;
    float *outputGradient_;
    float *inputGradient_;
    float *targetData_;

    void initializeNetwork();
    void allocateMemory();
    void initializeData();
    void runWarmup();
    double timeForwardPass();
    double timeBackwardInputPass();
    double timeBackwardParamsPass();
    void cleanup();
}; 