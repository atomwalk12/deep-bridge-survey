#include "benchmark/benchmark.hpp"

int main() {
    CUDNNBenchmark benchmark("cudnn/gpu/network_config.txt");
    benchmark.runBenchmark();
    return 0;
} 