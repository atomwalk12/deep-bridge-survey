
The report is available in [here](docs/report.pdf) and the report [here](docs/poster.pdf).

![image](https://github.com/user-attachments/assets/8e0ed1f4-09ff-42f5-a1b6-4dc73542d43a)

# Table of Contents

- [Table of Contents](#table-of-contents)
- [Distributed Training with Docker](#distributed-training-with-docker)
  - [Prerequisites](#prerequisites)
  - [Running the code](#running-the-code)
  - [Training script](#training-script)
- [Simple cuDNN network](#simple-cudnn-network)
  - [Dependencies](#dependencies)
  - [Building and running the Code](#building-and-running-the-code)
  - [Code walkthrough](#code-walkthrough)

# Distributed Training with Docker

## Prerequisites

To run this, you'll need to install the NVIDIA Container Toolkit by following the [official installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## Running the code

Key parameters:

- The `--nnodes` flag represents the number of machines communicating over a (simulated) network
- The `--nproc_per_node` flag stands for the number of GPUs on a single machine.

To simulate 2 GPUs (processes) on a single machine (node) run:

```bash
# Start the containers
docker-compose -f docker-compose-multi-process.yaml up -d

# Enter the container and login to wandb
docker-compose -f docker-compose-multi-process.yaml exec node bash
wandb login

# Single node, simulate two GPUs using two processes.
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=456 --rdzv_backend=c10d --rdzv_endpoint=node:48123 train.py --no_checkpoint
```

Also, you can simulate training over multiple machines with a single GPU for each:

```bash
# Start the containers
docker-compose -f docker-compose-multi-node.yaml up -d

# Open terminal 1 and login to wandb:
docker-compose -f docker-compose-multi-node.yaml exec node0 bash
wandb login

# Open terminal 2
# Note: the wandb query is not required for node1, as we log data through node0
docker-compose -f docker-compose-multi-node.yaml exec node1 bash

# Run the training script in each terminal
torchrun --nnodes=2 --nproc_per_node=1 --rdzv_id=456 --rdzv_backend=c10d --rdzv_endpoint=node0:48123 train.py --no_checkpoint
```

## Training script

The main training file is [train.py](./distributed/train.py).

# Simple cuDNN network

## Dependencies

1. Install the NVIDIA Container Toolkit by following the [official installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

2. Check your CUDA version by running

   ```bash
   nvidia-smi
   ```

3. Update the image tag in [docker-compose.yaml](cudnn/docker-compose.yaml) to match your GPU's CUDA version.

   - Current default: `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` (compatible with 40xx series GPUs)
   - Find your compatible image tag at [NVIDIA CUDA Docker Hub](https://hub.docker.com/r/nvidia/cuda/tags)

## Building and running the Code

Run the following to compile and run the code:

```bash
cd cudnn
docker-compose up -d # start the cotaniner
docker-compose exec cuda_dev bash # enter the container
./compile.sh
./runner
```

The docker-compose commands are optional. To execute the code without Docker, simply run `compile.sh` and `runner.sh` ignoring the docker commands. You'll need to have 

## Code walkthrough

- `Network` [@gpu/network.cu](./cudnn/gpu/network.cu): It stores an array of layers, manages forward/backward passes, and updates weights.

- `ConvolutionLayer` [@gpu/conv_layer.cu](./cudnn/gpu/conv_layer.cu): Implements 2D convolution operations using cuDNN

- `FCLayer` [@gpu/fc_layer.cu](./cudnn/gpu/fc_layer.cu): Implements fully connected layers using cuBLAS

- `ReLU` [@gpu/relu.cu](./cudnn/gpu/relu.cu): Implements the ReLU activation function using custom CUDA kernels

- `MSELoss` [@gpu/loss.cu](./cudnn/gpu/loss.cu): Implements Mean Squared Error loss computation and gradients

- `CostHistory` [@gpu/utils.cu](./cudnn/gpu/utils.cu): Tracks and visualizes training loss over time

Below is a simplified example implementation of a neural network training process. Check [toy_network.cu](./cudnn/gpu/toy_network.cu) for the complete code:

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
