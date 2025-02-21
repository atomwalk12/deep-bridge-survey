# Simple cuDNN network

This repository contains benchmarks for GPU training using cuDNN and examples for simulating distributed training using Docker Swarm.

## Prerequisites

1. Install the NVIDIA Container Toolkit by following the [official installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
2. Check your CUDA version by running

   ```bash
   nvidia-smi
   ```
3. Update the image tag in [docker-compose.yaml](cudnn/docker-compose.yaml) to match your GPU's CUDA version.
   - Current default: `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` (compatible with 40xx series GPUs)
   - Find your compatible image tag at [NVIDIA CUDA Docker Hub](https://hub.docker.com/r/nvidia/cuda/tags)

## Getting Started

### Building and Running the Code

Run the following to compile and run the code:

```bash
cd cudnn
docker-compose up -d # start the cotaniner
docker-compose exec cuda_dev bash # enter the container
./compile.sh
./runner
```

## Code walkthrough

- `Network`: It stores an array of layers, manages forward/backward passes, and updates weights.

- `ConvolutionLayer`: Implements 2D convolution operations using cuDNN
- `FCLayer`: Implements fully connected layers using cuBLAS
- `ReLU`: Implements the ReLU activation function using custom CUDA kernels

- `MSELoss`: Implements Mean Squared Error loss computation and gradients

- `CostHistory`: Tracks and visualizes training loss over time

### Example

Below is a simplified example implementation of a neural network training. Check [toy_network.cu](./cudnn/gpu/toy_network.cu) for the complete code:

```c++
// ==============================
// Initialization
// ==============================
float input_data[BATCH_SIZE][IN_CHANNELS][INPUT_HEIGHT][INPUT_WIDTH] = {
    {   // batch 0
        {   // channel 0
            {1.0f, 0.0f, 1.0f},  // row 0
            {0.0f, 1.0f, 0.0f},  // row 1
            {1.0f, 0.0f, 1.0f}   // row 2
        }
    }
};


// Use one-hot encoding where 1 represents the correct class
float* target_data;
int NUM_CLASSES = 10
int OUTPUT_SIZE = 1 * NUM_CLASSES; // batch_size * NUM_CLASSES

for (int i = 0; i < OUTPUT_SIZE; i++) {
    target_data[i] = (i == 0) ? 1.0f : 0.0f;
}

// ==============================
// Create the network
// ==============================
Network model(cudnn, BATCH_SIZE, NUM_CLASSES, INPUT_WIDTH, INPUT_HEIGHT, IN_CHANNELS);

// AddConvLayer input is in the form (out_channels, kernel_size, stride, padding)
model.addConvLayer(16, 3, 1, 1);
model.addConvLayer(32, 3, 1, 1);
model.addConvLayer(64, 3, 1, 1);

// Here we have input as (in_neurons, out_neurons)
model.addFCLayer(model.getFlattenedSize(), 512);
model.addFCLayer(512, 128);
model.addFCLayer(128, 10); // output has 10 classes

float* output_gradient = model.createDummyGradient(output_data);
float* input_gradient;

cudaMallocManaged(&input_gradient, INPUT_GRADIENT_SIZE * sizeof(float));
cudaDeviceSynchronize();

// ==============================
// Training
// ==============================
MSELoss loss;

for (int i = 0; i < NUM_ITERATIONS; i++) {
    model.zeroGradients();
    model.forward(input_data, output_data);

    float loss_value = loss.compute(output_data, target_data, OUTPUT_SIZE);

    if (i % 10 == 0) cost_history_add(&cost_history, loss_value);

    // Backwards using mean-squared-error
    loss.backward(output_data, target_data, output_gradient, OUTPUT_SIZE);

    // We do the backward step separately w.r.t the input then parameters
    model.backwardInput(input_gradient, output_gradient);
    model.backwardParams(input_data, output_gradient);

    // Learning step with small learning rate
    model.updateWeights(0.001f);

    printf("Iteration %d, Loss: %f\n", i, loss_value);
}

plot_cost_ascii(&cost_history);
/*
Cost Function Over Epochs
2.5157 ┐
3
|                                                           
| *                                                         
|                                                           
|   *                                                       
|     *                                                     
|       *                                                   
|         *                                                 
|           *                                               
|             *                                             
|               * *                                         
|                   *                                       
|                     * *                                   
|                         * *                               
|                             ** *                          
|                                  * * *                    
|                                        * * * *            
|                                                * * * * *  
------------------------------------------------------------
  0.4410 ┴────────────────────────────────────────────────── 30 epochs
*/
```

