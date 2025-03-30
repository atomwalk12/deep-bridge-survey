The report is available [here](docs/report.pdf) and the poster [here](docs/poster.pdf).

![image](https://github.com/user-attachments/assets/e1982144-378b-45cf-a6ab-2628169c5565)


<!-- TABLE OF CONTENTS -->

<details>
  <summary>Table of Contents</summary>
  <ol>
      <li>
      <a href="#resources">Resources</a>
      <ul>
        <li><a href="#conducting-a-survey">Conducting a survey</a></li>
        <li><a href="#similar-surveys">Similar surveys</a></li>
        <li><a href="#dnn-studies">DNN Studies</a></li>
        <li><a href="#gpu-studies">GPU Studies</a></li>
      </ul>
    </li>
    <li>
      <a href="#distributed-training-with-docker">Distributed Training with Docker</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#running-the-code">Running the code</a></li>
        <li><a href="#training-script">Training script</a></li>
      </ul>
    </li>
    <li>
      <a href="#simple-cudnn-network">Simple cuDNN Network</a>
      <ul>
        <li><a href="#dependencies">Dependencies</a></li>
        <li><a href="#building-and-running-the-code">Building and running the code</a></li>
        <li><a href="#code-walkthrough">Code walkthrough</a></li>
      </ul>
    </li>
  </ol>
</details>

# Resources

## Conducting a survey

- :fire: :fire: :fire:   **Artificial intelligence for literature reviews: opportunities and challenges**, Springer [[paper](https://doi.org/10.1007/s10462-024-10902-3)]


- **Guidelines for performing Systematic Literature Reviews in Software Engineering**, None [[paper](https://legacyfileshare.elsevier.com/promis_misc/525444systematicreviewsguide.pdf)]
- **Lessons from applying the systematic literature review process within the software engineering domain**, Elsevier [[paper](https://linkinghub.elsevier.com/retrieve/pii/S016412120600197X)]
- **Procedures for Performing Systematic Reviews**, ResearchGate [[paper](https://www.researchgate.net/publication/228756057_Procedures_for_Performing_Systematic_Reviews)]
- **Reporting systematic reviews: Some lessons from a tertiary study**, ScienceDirect [[paper](https://www.sciencedirect.com/science/article/pii/S0950584916303548)]
- :fire: :fire: **Sustainable systematic literature reviews**, ScienceDirect [[paper](https://www.sciencedirect.com/science/article/pii/S0950584924001563)]

## Similar surveys

- **Machine Learning and Deep Learning frameworks and libraries for large-scale data mining: a survey**, Springer [[paper](https://doi.org/10.1007/s10462-018-09679-z)]
- **From distributed machine to distributed deep learning: a comprehensive survey**, Springer [[paper](https://doi.org/10.1186/s40537-023-00829-x)]
- :fire: **A Hitchhiker’s Guide On Distributed Training of Deep Neural Networks**, arXiv:1810.11787 [[paper](http://arxiv.org/abs/1810.11787)]
- **A Systematic Review of Distributed Deep Learning Frameworks for Big Data**, Springer [[paper](https://doi.org/10.1007/978-3-031-13832-4_21)]

## DNN Studies

**Data:**

-  **TensorFlow**, arXiv:1603.04467 [[paper](http://arxiv.org/abs/1603.04467)]
- **BytePS**, arXiv:None [[paper](https://www.usenix.org/system/files/osdi20-jiang.pdf)]
- :fire: :fire: :fire:   **HuggingFace's Transformers**, arXiv:1910.03771 [[paper](http://arxiv.org/abs/1910.03771)]
- :fire:  **Pytorch Lightning**, official [[docs](https://lightning.ai/docs/overview/)]
- **FairScale**, official [[docs](https://fairscale.readthedocs.io/en/latest/)]
- **Amazon SageMaker Platform**, official [[docs](https://docs.aws.amazon.com/sagemaker/)]
- **Microsoft AzureML Platform**, official [[docs](https://learn.microsoft.com/en-us/azure/machine-learning/?view=azureml-api-2)]
- :fire: :fire:  **Google Vertex AI Platform**, official [[docs](https://cloud.google.com/vertex-ai/docs)]

**Hybrid:**

- **MXNet**, SemanticScholar [[paper](https://www.semanticscholar.org/paper/MXNet%3A-A-Flexible-and-Efficient-Machine-Learning-Chen-Li/62df84d6a4d26f95e4714796c2337c9848cc13b5)]
- :fire: :fire: **PyTorch Distributed**, arXiv:2006.15704 [[paper](http://arxiv.org/abs/2006.15704)]
- **Colossal AI**, arXiv:2110.14883 [[paper](http://arxiv.org/abs/2110.14883)]
- **Ray**, arXiv:1712.05889 [[paper](http://arxiv.org/abs/1712.05889)]
- :fire: **DeepSpeed**, ACM [[paper](https://dl.acm.org/doi/10.1145/3394486.3406703)]
- **Horovod**, arXiv:1802.05799 [[paper](http://arxiv.org/abs/1802.05799)]
- :fire: **Jax**, GoogleResearch [[paper](https://research.google/pubs/compiling-machine-learning-programs-via-high-level-tracing/)]

**Pipeline:**

- **GPipe**, arXiv:1811.06965 [[paper](http://arxiv.org/abs/1811.06965)]

**Model:**

- :fire: **GShard**, arXiv:2006.16668 [[paper](http://arxiv.org/abs/2006.16668)]
- :fire: **Megatron-LM**, arXiv:1909.08053 [[paper](http://arxiv.org/abs/1909.08053)]

---

## GPU Studies

**NVIDIA:**

- :fire: :fire: **cuBLAS**, official [[docs](https://developer.nvidia.com/cublas)]
- :fire: :fire: **cuDNN**, arXiv:1410.0759 [[paper](http://arxiv.org/abs/1410.0759)]
- **NCCL**, official [[docs](https://developer.nvidia.com/nccl)]
- **CUTLASS**, official [[github](https://github.com/NVIDIA/cutlass)]
- :fire: **CUDA Toolkit Samples**, official [[examples](https://github.com/NVIDIA/cuda-samples)]
- **CUDA.jl**, official [[github](https://github.com/JuliaGPU/CUDA.jl)]
- :fire: **CuPy**, SemanticScholar [[paper](https://www.semanticscholar.org/paper/CuPy-%3A-A-NumPy-Compatible-Library-for-NVIDIA-GPU-Okuta-Unno/a59da4639436f582e483347a4833e7659fd3e598)]
- **Numba**, official [[github](https://github.com/numba/numba)]

**AMD:**

- :fire: **rocBLAS**, official [[github](https://github.com/ROCm/rocBLAS)]
- :fire: **MIOpen**, official [[github](https://github.com/ROCm/MIOpen)]
- **HIP**, official [[github](https://github.com/ROCm/HIP)]
- **RCCL**, official [[github](https://github.com/ROCm/rccl)]
- **AMD Lab Notes**, official [[examples](https://github.com/amd/amd-lab-notes)]
- **AMDGPU.jl**, official [[github](https://github.com/JuliaGPU/AMDGPU.jl)]

**Intel:**

- **oneMKL**, official [[docs](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-documentation.html)]
- :fire: **oneDNN**, official [[github](https://github.com/oneapi-src/oneDNN)]
- **oneCCL**, official [[github](https://github.com/uxlfoundation/oneCCL)]
- **oneAPI.jl**, official [[github](https://github.com/JuliaGPU/oneAPI.jl)]
- **Intel Compute Samples**, official [[examples](https://github.com/intel/compute-samples)]

**Cross-Platform:**

- **OpenCL**, official-docs [[github](https://github.com/KhronosGroup/OpenCL-Docs)]
- **SYCL**, official-docs [[github](https://github.com/KhronosGroup/SYCL-Docs)]
- **Kokkos**, official [[github](https://github.com/kokkos/kokkos)]

**NN Libraries:**

- :fire: **Caffe**, arXiv:1408.5093 [[paper](http://arxiv.org/abs/1408.5093)]
- :fire: **Cuda-Convnet**, CodeArchive [[paper](https://code.google.com/archive/p/cuda-convnet/)]
- **Pylearn2**, arXiv:1308.4214 [[paper](http://arxiv.org/abs/1308.4214)]
- **Torch7**, collobert [[paper](https://ronan.collobert.com/pub/matos/2011_torch7_nipsw.pdf)]


# Distributed Training with Docker

## Prerequisites

To run this, you'll need to install the NVIDIA Container Toolkit by following the [official installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## Running the code

Key parameters:

- The `--nnodes` flag represents the number of machines communicating over a (simulated) network
- The `--nproc_per_node` flag stands for the number of GPUs on a single machine.

To simulate 2 GPUs (processes) on a single machine (node) run:

```bash
cd distributed
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
cd distributed
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
./compile_benchmark.sh # or ./compile_toynetwork.sh
./runner
# Sample output for benchmark:
Average forward pass time: 0.185160 ms
Average backward input pass time: 0.161860 ms
Average backward params pass time: 0.030010 ms
Total time: 0.377030 ms
```

The docker-compose commands are optional. To execute the code without Docker, simply run `compile.sh` and `runner.sh` ignoring the docker commands.

## Code walkthrough

- `Network` [@gpu/core/network.cu](./cudnn/gpu/core/network.cu): It stores an array of layers, manages forward/backward passes, and updates weights.

- `ConvolutionLayer` [@gpu/layers/conv_layer.cu](./cudnn/gpu/layers/conv_layer.cu): Implements 2D convolution operations using cuDNN

- `FCLayer` [@gpu/layers/fc_layer.cu](./cudnn/gpu/layers/fc_layer.cu): Implements fully connected layers using cuBLAS

- `ReLU` [@gpu/layers/relu.cu](./cudnn/gpu/layers/relu.cu): Implements the ReLU activation function using custom CUDA kernels

- `MSELoss` [@gpu/loss/loss.cu](./cudnn/gpu/loss/loss.cu): Implements Mean Squared Error loss computation and gradients

- `CostHistory` [@gpu/core/utils.cu](./cudnn/gpu/core/utils.cu): Tracks and visualizes training loss over time

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

## Benchmarking

Evaluation results for cuDNN and PyTorch networks, configured identically according to [network_config.txt](./cudnn/gpu/network_config.txt) and [network_config.json](./pytorch/network_config.json), are presented below:


| Framework | Average Forward Pass Time (ms) | Average Backward Input Pass Time (ms) | Average Backward Params Pass Time (ms) | Total Time (ms) |
| --------- | ------------------------------ | ------------------------------------- | -------------------------------------- | --------------- |
| cuDNN     | 0.206760                       | 0.214760                              | 0.028600                               | 0.450120        |
| PyTorch   | 0.137913                       | 0.112698                              | 0.198538                               | 0.449149        |


The files to replicate the results are available in the [pytorch](./pytorch/benchmark.py) and [cudnn](./cudnn/gpu/run_benchmark.cu) directories.
