#!/bin/bash

SOURCES=(
    gpu/benchmark.cu
    gpu/alexnet.cu
    gpu/conv_layer.cu
    gpu/utils.cu
    gpu/loss.cu
    gpu/fc_layer.cu
    gpu/relu.cu
)

FLAGS="-g -G -O0 -arch=sm_89 --compiler-options -fPIC"
LIBS="-lcudnn -lcublas"

nvcc $FLAGS "${SOURCES[@]}" -o runner $LIBS
