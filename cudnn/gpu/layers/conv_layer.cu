#include "../core/utils.h"
#include "conv_layer.h"
#include <stdio.h>
#include <cublas_v2.h>
#include <random>
#include <cmath>

ConvolutionLayer::ConvolutionLayer(cudnnHandle_t &cudnn_handle,
                                   int input_width,
                                   int input_height,
                                   int batch_size,
                                   int in_channels,
                                   int out_channels,
                                   int kernel_size,
                                   int stride,
                                   int padding)
    : Layer(cudnn_handle),
      batch_size(batch_size),
      in_channels(in_channels),
      out_channels(out_channels),
      kernel_size(kernel_size),
      stride(stride),
      padding(padding),
      input_height(input_height),
      input_width(input_width)
{

    createDescriptors();
    calculateOutputDimensions();

    // Initialize ReLU with the total size of the output
    int total_elements = batch_size * out_channels * output_height * output_width;
    relu = new ReLU(total_elements);

    // Create cuBLAS handle and check for success
    cublasStatus_t cublas_status = cublasCreate(&cublas_handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS initialization failed: %d\n", cublas_status);
        exit(1);
    }
}

void ConvolutionLayer::createDescriptors()
{
    cudnnCreateTensorDescriptor(&input_descriptor);
    cudnnSetTensor4dDescriptor(
        input_descriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size, in_channels, input_height, input_width);

    cudnnCreateFilterDescriptor(&filter_descriptor);
    cudnnSetFilter4dDescriptor(
        filter_descriptor,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        out_channels, in_channels, kernel_size, kernel_size);

    cudnnCreateConvolutionDescriptor(&conv_descriptor);
    cudnnSetConvolution2dDescriptor(
        conv_descriptor,
        padding,
        padding,
        stride,
        stride,
        1, // dilation_h
        1, // dilation_w
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT);

    int out_n, out_c, out_h, out_w;
    cudnnGetConvolution2dForwardOutputDim(
        conv_descriptor,
        input_descriptor,
        filter_descriptor,
        &out_n,
        &out_c,
        &out_h,
        &out_w);

    output_height = out_h;
    output_width = out_w;

    cudnnCreateTensorDescriptor(&output_descriptor);
    cudnnSetTensor4dDescriptor(
        output_descriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size, out_channels, output_height, output_width);

    // Initialize weights and biases
    size_t weight_size = getWeightSize();
    cudaMallocManaged(&weights, weight_size * sizeof(float));
    cudaMallocManaged(&weight_gradients, weight_size * sizeof(float));

    std::random_device rd;
    std::mt19937 gen(rd());
    float std = sqrt(2.0f / (in_channels * kernel_size * kernel_size));
    std::normal_distribution<float> distribution(0.0f, std);

    for (size_t i = 0; i < weight_size; i++)
    {
        weights[i] = distribution(gen);
    }

    debugDescriptor("Input", input_descriptor);
    debugDescriptor("Output", output_descriptor);
    debugFilterDescriptor(filter_descriptor);
    debugTensorValues("weights", weights, 10);

    // Get workspace size needed for the selected algorithm
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    workspace_size = 0;
    cudnnGetConvolutionForwardWorkspaceSize(
        cudnn,
        input_descriptor,
        filter_descriptor,
        conv_descriptor,
        output_descriptor,
        algo,
        &workspace_size);

    // Allocate workspace memory
    workspace = nullptr;
    if (workspace_size > 0)
    {
        cudaMalloc(&workspace, workspace_size);
    }
}

void ConvolutionLayer::forward(float *input, float *output)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Allocate temporary buffer for convolution output before ReLU
    float *conv_output;

    size_t output_size = batch_size * out_channels * output_height * output_width * sizeof(float);

    cudaMallocManaged(&conv_output, output_size);

    // Perform convolution using the pre-allocated workspace
    cudnnStatus_t status = cudnnConvolutionForward(
        cudnn,
        &alpha,
        input_descriptor,
        input,
        filter_descriptor,
        weights,
        conv_descriptor,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
        workspace,
        workspace_size,
        &beta,
        output_descriptor,
        conv_output);
    cudaDeviceSynchronize();

    if (status != CUDNN_STATUS_SUCCESS)
    {
        printf("CUDNN forward failed: %s\n", cudnnGetErrorString(status));
        exit(1);
    }

    // Apply ReLU activation
    relu->forward(conv_output, output); // Output first, then input

    // Free temporary buffer
    cudaFree(conv_output);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}

void ConvolutionLayer::destroyDescriptors()
{
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyFilterDescriptor(filter_descriptor);
    cudnnDestroyConvolutionDescriptor(conv_descriptor);
    cudaFree(weights);
    cudaFree(bias);
    cudaFree(weight_gradients);
}

void ConvolutionLayer::backwardInput(float *input_gradient, float *output_gradient)
{
    // First apply ReLU backward
    float *relu_gradient_output;
    size_t output_size = batch_size * out_channels * output_height * output_width * sizeof(float);
    cudaMallocManaged(&relu_gradient_output, output_size);

    relu->backward(relu_gradient_output, output_gradient);

    // Then proceed with normal conv backward using the modified gradient
    debugTensorValues("output gradient", relu_gradient_output, 10);

    // Verify pointers
    if (input_gradient == nullptr || relu_gradient_output == nullptr || weights == nullptr)
    {
        printf("Error: Null pointer in backwardInput\n");
        exit(1);
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Get workspace size needed
    size_t workspace_size = 0;
    cudnnStatus_t status = cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnn,
        filter_descriptor,
        output_descriptor,
        conv_descriptor,
        input_descriptor,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
        &workspace_size);

    if (status != CUDNN_STATUS_SUCCESS)
    {
        printf("Error getting workspace size: %s\n", cudnnGetErrorString(status));
        exit(1);
    }

    // Allocate workspace
    void *workspace = nullptr;
    if (workspace_size > 0)
    {
        cudaError_t err = cudaMalloc(&workspace, workspace_size);
        if (err != cudaSuccess)
        {
            printf("Workspace allocation failed: %s\n", cudaGetErrorString(err));
            exit(1);
        }
    }

    // Backward pass with different algorithm
    // Implements: dL/dx = (dL/dy) @ w
    status = cudnnConvolutionBackwardData(
        cudnn,
        &alpha,
        filter_descriptor,
        weights,
        output_descriptor,
        relu_gradient_output, // Use the gradient after ReLU backward
        conv_descriptor,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
        workspace,
        workspace_size,
        &beta,
        input_descriptor,
        input_gradient);

    if (status != CUDNN_STATUS_SUCCESS)
    {
        printf("Backward data failed: %s\n", cudnnGetErrorString(status));
        exit(1);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    if (workspace)
    {
        cudaFree(workspace);
    }

    cudaFree(relu_gradient_output);
    debugTensorValues("input gradient", input_gradient, 10);
}

void ConvolutionLayer::backwardParams(float *input, float *output_gradient)
{
    debugTensorValues("input", input, 10);
    debugTensorValues("output gradient", output_gradient, 10);

    if (input == nullptr || output_gradient == nullptr || weight_gradients == nullptr)
    {
        printf("Error: Null pointer passed to backwardParams\n");
        return;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Implements: dL/dW = x ⊗ (dL/dy)
    cudnnStatus_t status = cudnnConvolutionBackwardFilter(
        cudnn,
        &alpha,
        input_descriptor,
        input,
        output_descriptor,
        output_gradient,
        conv_descriptor,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
        nullptr,
        0,
        &beta,
        filter_descriptor,
        weight_gradients);

    if (status != CUDNN_STATUS_SUCCESS)
    {
        printf("Backward filter failed: %s\n", cudnnGetErrorString(status));
        exit(1);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error after backward filter: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    debugTensorValues("weight gradients", weight_gradients, 10);
}

void ConvolutionLayer::zeroGradients()
{
    cudaMemset(weight_gradients, 0, getWeightSize() * sizeof(float));
}

ConvolutionLayer::~ConvolutionLayer()
{
    destroyDescriptors();

    cublasDestroy(cublas_handle);
    if (workspace)
    {
        cudaFree(workspace);
    }
    delete relu;
}
