#!/bin/bash

SOURCES=(
    gpu/benchmark.cu
    gpu/core/network.cu
    gpu/layers/conv_layer.cu
    gpu/core/utils.cu
    gpu/loss/loss.cu
    gpu/layers/fc_layer.cu
    gpu/layers/relu.cu
)

FLAGS="-g -G -O0 -arch=sm_89 --compiler-options -fPIC"
LIBS="-lcudnn -lcublas"

nvcc $FLAGS "${SOURCES[@]}" -o runner $LIBS
