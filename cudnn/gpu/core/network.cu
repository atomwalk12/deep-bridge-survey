#include "network.h"
#include <cstdlib>
#include "utils.h"
#include <string>

Network::Network(cudnnHandle_t &handle, int batch_size, int num_classes_, int initial_width, int initial_height, int initial_channels)
    : cudnn(handle),
      batch_size_(batch_size),
      num_classes_(num_classes_),
      current_width_(initial_width),
      current_height_(initial_height),
      current_channels_(initial_channels)
{
    cublasStatus_t status = cublasCreate(&cublas);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS initialization failed: %d\n", status);
        exit(1);
    }
}

void Network::forward(float *inp, float *out)
{
    float *current_input = inp;

    for (size_t i = 0; i < layers.size(); i++)
    {
        float *current_output = (i == layers.size() - 1) ? out : layer_outputs[i];
        layers[i]->forward(current_input, current_output);
        current_input = current_output;
    }
}

float *Network::createDummyGradient(float *output)
{
    // Create a gradient of ones, similar to PyTorch's ones_like
    size_t output_dim = batch_size_ * num_classes_;
    float *gradient;
    cudaMallocManaged(&gradient, output_dim * sizeof(float));

    for (size_t i = 0; i < output_dim; i++)
    {
        gradient[i] = 1.0f;
    }
    return gradient;
}

void Network::backwardInput(float *inp_grad, float *out_grad)
{
    float *current_output_grad = out_grad;

    // Backward pass through layers in reverse order
    for (int i = layers.size() - 1; i >= 0; i--)
    {
        // Determine where to store the computed gradient
        float *current_input_grad;
        if (i == 0)
        {
            current_input_grad = inp_grad;
        }
        else
        {
            current_input_grad = gradient_outputs[i - 1];
        }

        // Compute gradients
        if (auto *conv_layer = dynamic_cast<ConvolutionLayer *>(layers[i]))
        {
            conv_layer->backwardInput(current_input_grad, current_output_grad);
        }
        else if (auto *fc_layer = dynamic_cast<FCLayer *>(layers[i]))
        {
            fc_layer->backwardInput(current_input_grad, current_output_grad);
        }

        // Update for next iteration
        current_output_grad = current_input_grad;
    }
}

void Network::backwardParams(float *inp, float *out_grad)
{
    float *current_input = inp;
    float *current_output_grad = out_grad;

    // Compute parameter gradients for each layer
    for (int i = layers.size() - 1; i >= 0; i--)
    {
        if (auto *conv_layer = dynamic_cast<ConvolutionLayer *>(layers[i]))
        {
            conv_layer->backwardParams(current_input, current_output_grad);
        }
        else if (auto *fc_layer = dynamic_cast<FCLayer *>(layers[i]))
        {
            fc_layer->backwardParams(current_input, current_output_grad);
        }

        if (i > 0)
        {
            current_input = layer_outputs[i - 1];
            current_output_grad = layer_outputs[i - 1];
        }
    }
}

void Network::updateWeights(float learning_rate)
{
    float lr = -learning_rate; // Negative because cublasSaxpy does addition

    for (int i = layers.size() - 1; i >= 0; i--)
    {
        ConvolutionLayer *conv = static_cast<ConvolutionLayer *>(layers[i]);
        // Update conv1 weights
        cublasSaxpy(cublas,
                    conv->getWeightSize(),
                    &lr,
                    conv->getWeightGradients(), 1,
                    conv->getWeights(), 1);

        if (ENABLE_DEBUG_OUTPUT)
        {
            checkWeightChanges(("Layer " + std::to_string(i)).c_str(), conv->getWeights(), conv->getWeightSize());
        }
    }
}

void Network::zeroGradients()
{
    for (Layer *layer : layers)
    {
        layer->zeroGradients();
    }
}

Network::~Network()
{

    cublasDestroy(cublas);

    for (Layer *layer : layers)
    {
        delete layer;
    }

    // Free forward pass outputs
    for (float *output : layer_outputs)
    {
        if (output != nullptr)
        {
            cudaFree(output);
        }
    }

    // Free gradient outputs
    for (float *gradient : gradient_outputs)
    {
        if (gradient != nullptr)
        {
            cudaFree(gradient);
        }
    }
}

void Network::addConvLayer(int out_channels, int kernel_size, int stride, int padding)
{
    ConvolutionLayer *conv_layer = new ConvolutionLayer(
        cudnn,
        current_width_,
        current_height_,
        batch_size_,
        current_channels_,
        out_channels,
        kernel_size,
        stride,
        padding);
    layers.push_back(conv_layer);

    // Update current dimensions for next layer
    current_width_ = conv_layer->getOutputWidth();
    current_height_ = conv_layer->getOutputHeight();
    current_channels_ = out_channels;

    size_t output_size = batch_size_ * current_channels_ * current_height_ * current_width_;

    float *layer_output;
    cudaMallocManaged(&layer_output, output_size * sizeof(float));
    layer_outputs.push_back(layer_output);

    float *gradient_output;
    cudaMallocManaged(&gradient_output, output_size * sizeof(float));
    gradient_outputs.push_back(gradient_output);
}

void Network::addFCLayer(int in_features, int out_features)
{
    layers.push_back(new FCLayer(cudnn, batch_size_, in_features, out_features));

    size_t output_size = batch_size_ * out_features * sizeof(float);

    float *layer_output;
    cudaMalloc(&layer_output, output_size);
    layer_outputs.push_back(layer_output);

    float *gradient_output;
    cudaMalloc(&gradient_output, output_size);
    gradient_outputs.push_back(gradient_output);
}
