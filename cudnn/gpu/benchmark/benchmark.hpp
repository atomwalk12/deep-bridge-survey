#pragma once

#include <cudnn.h>
#include "../core/network.h"
#include "../loss/loss.h"
#include "../core/utils.h"

class CUDNNBenchmark {
public:
    CUDNNBenchmark(
        std::string config_path_,
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
    int batchSize_;
    int numClasses_;
    int inChannels_;
    int inputHeight_;
    int inputWidth_;
    int warmupIterations_;
    int benchmarkIterations_;
    const std::string config_path_;

    // Derived sizes
    int inputSize_;
    int outputSize_;
    int inputGradientSize_;

    // CUDNN handle
    cudnnHandle_t cudnn_;
    Network* model_;
    
    // Data pointers
    float *inputData_;
    float *outputData_;
    float *outputGradient_;
    float *inputGradient_;
    float *targetData_;

    void initializeNetwork(
        const std::vector<std::vector<int>>& convLayers,
        const std::vector<std::vector<int>>& fcLayers
    );
    void allocateMemory();
    void initializeData();
    void runWarmup();
    double timeForwardPass();
    double timeBackwardInputPass();
    double timeBackwardParamsPass();
    void cleanup();
}; 