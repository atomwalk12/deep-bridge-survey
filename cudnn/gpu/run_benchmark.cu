#include "benchmark/benchmark.hpp"

int main() {
    CUDNNBenchmark benchmark;
    benchmark.runBenchmark();
    return 0;
} 