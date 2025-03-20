#include "benchmark/benchmark.hpp"

int main(int argc, char **argv) {
    // Default configuration file path
    std::string config_path = "gpu/network_config.txt"; // Assume the program is running from the cudnn folder
    
    if (argc > 1) {
        config_path = argv[1]; // if command line argument is provided
    }
    CUDNNBenchmark benchmark(config_path);
    benchmark.runBenchmark();
    return 0;
} 