# CUDA Training Benchmarks

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

