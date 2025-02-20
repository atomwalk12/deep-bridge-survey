#include "utils.h"
#include <stdio.h>

// Debug configuration
bool ENABLE_DEBUG_OUTPUT = false;  // Definition with default value

void debugDescriptor(const char* name, cudnnTensorDescriptor_t desc) {
    if (!ENABLE_DEBUG_OUTPUT) return;
    cudnnDataType_t dtype;
    int n, c, h, w;
    int stride_n, stride_c, stride_h, stride_w;
    
    cudnnGetTensor4dDescriptor(desc,
        &dtype,
        &n, &c, &h, &w,
        &stride_n, &stride_c, &stride_h, &stride_w);
        
    printf("%s descriptor:\n", name);
    printf("  Dimensions: %dx%dx%dx%d\n", n, c, h, w);
    printf("  Strides: %d,%d,%d,%d\n", stride_n, stride_c, stride_h, stride_w);
    fflush(stdout);
}

void debugFilterDescriptor(cudnnFilterDescriptor_t desc) {
    if (!ENABLE_DEBUG_OUTPUT) return;
    cudnnDataType_t dtype;
    cudnnTensorFormat_t format;
    int k, c, h, w;
    
    cudnnGetFilter4dDescriptor(desc,
        &dtype,
        &format,
        &k, &c, &h, &w);
        
    printf("Filter descriptor:\n");
    printf("  Dimensions: %dx%dx%dx%d\n", k, c, h, w);
    fflush(stdout);
}

void debugTensorValues(const char* label, float* device_ptr, int count) {
    if (!ENABLE_DEBUG_OUTPUT) return;
    
    float debug_values[10];  // Static size for simplicity
    int print_count = std::min(count, 10);  // Print at most 10 values
    
    cudaError_t err = cudaMemcpy(debug_values, device_ptr, 
                                print_count * sizeof(float), 
                                cudaMemcpyDeviceToHost);
    
    if (err != cudaSuccess) {
        printf("Error copying %s values: %s\n", 
               label, cudaGetErrorString(err));
        return;
    }

    printf("First %d %s values: ", print_count, label);
    for(int i = 0; i < print_count; i++) {
        printf("%.4f ", debug_values[i]);
    }
    printf("\n");
    fflush(stdout);
}

void checkWeightChanges(const char* label, float* device_weights, int size) {

    static float prev_sum = 0.0f;  // Keep track of previous sum
    
    // Copy weights to host
    float* host_weights = new float[size];
    cudaMemcpy(host_weights, device_weights, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Compute simple statistics
    float sum = 0.0f;
    for(int i = 0; i < size; i++) {
        sum += host_weights[i];
    }
    
    printf("%s - Weight sum: %.4f (change: %.4f)\n", 
           label, sum, sum - prev_sum);
    fflush(stdout);
    prev_sum = sum;
    
    delete[] host_weights;
}

void debugMatrixLayout(const char* label, float* device_ptr, 
                      int rows, int cols, int max_rows, int max_cols) {
    printf("\nMatrix Layout Debug for %s:\n", label);
    printf("Full dimensions: [%d × %d]\n", rows, cols);
    
    // Limit the size we're copying
    int display_rows = std::min(rows, max_rows);
    int display_cols = std::min(cols, max_cols);
    
    float* host_data = new float[rows * cols];
    cudaMemcpy(host_data, device_ptr, rows * cols * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    printf("\nAssuming Row-Major Layout:\n");
    for(int i = 0; i < display_rows; i++) {
        printf("Row %d: ", i);
        for(int j = 0; j < display_cols; j++) {
            printf("%8.4f ", host_data[i * cols + j]);
        }
        printf("\n");
    }
    
    printf("\nAssuming Column-Major Layout:\n");
    for(int i = 0; i < display_rows; i++) {
        printf("Row %d: ", i);
        for(int j = 0; j < display_cols; j++) {
            printf("%8.4f ", host_data[i + j * rows]);
        }
        printf("\n");
    }
    printf("\n");
    fflush(stdout);
    
    delete[] host_data;
} 


void cost_history_init(CostHistory *history) {
    history->count = 0;
    history->min = 1e9;
    history->max = -1e9;
}

void cost_history_add(CostHistory *history, float value) {
    if (history->count < MAX_HISTORY) {
        history->values[history->count++] = value;
        if (value < history->min) history->min = value;
        if (value > history->max) history->max = value;
    }
}

void plot_cost_ascii(CostHistory *history) {
    // This function was inspired from:
    // https://github.com/karam-koujan/mini-pytorch/
    char graph[GRAPH_HEIGHT][GRAPH_WIDTH + 1];
    float range = history->max - history->min;
    
    // Initialize graph with spaces
    for (int i = 0; i < GRAPH_HEIGHT; i++) {
        memset(graph[i], ' ', GRAPH_WIDTH);
        graph[i][GRAPH_WIDTH] = '\0';
    }
    
    // Draw axis
    for (int i = 0; i < GRAPH_HEIGHT; i++) {
        graph[i][0] = '|';
    }
    for (int i = 0; i < GRAPH_WIDTH; i++) {
        graph[GRAPH_HEIGHT-1][i] = '-';
    }
    
    // Plot points with improved x-axis distribution
    float min_y[GRAPH_WIDTH];  // Track minimum value for each x position
    for (int i = 0; i < GRAPH_WIDTH; i++) {
        min_y[i] = history->max;
    }
    
    // First pass: find minimum values for each x position
    for (int i = 0; i < history->count; i++) {
        int x = (int)((float)i / history->count * (GRAPH_WIDTH - 2)) + 1;
        if (x < GRAPH_WIDTH && history->values[i] < min_y[x]) {
            min_y[x] = history->values[i];
        }
    }
    
    // Second pass: plot the points
    for (int x = 1; x < GRAPH_WIDTH; x++) {
        if (min_y[x] != history->max) {
            float normalized = (min_y[x] - history->min) / range;
            int y = GRAPH_HEIGHT - 2 - (int)(normalized * (GRAPH_HEIGHT - 3));
            if (y >= 0 && y < GRAPH_HEIGHT) {
                graph[y][x] = '*';
            }
        }
    }
    
    // Connect adjacent points with lines
    for (int x = 1; x < GRAPH_WIDTH - 1; x++) {
        if (min_y[x] != history->max && min_y[x+1] != history->max) {
            float norm1 = (min_y[x] - history->min) / range;
            float norm2 = (min_y[x+1] - history->min) / range;
            int y1 = GRAPH_HEIGHT - 2 - (int)(norm1 * (GRAPH_HEIGHT - 3));
            int y2 = GRAPH_HEIGHT - 2 - (int)(norm2 * (GRAPH_HEIGHT - 3));
            
            // Draw connecting line
            int start_y = (y1 < y2) ? y1 : y2;
            int end_y = (y1 < y2) ? y2 : y1;
            for (int y = start_y + 1; y < end_y; y++) {
                if (y >= 0 && y < GRAPH_HEIGHT) {
                    graph[y][x] = '|';
                }
            }
        }
    }
    
    // Print graph with axis labels
    printf("\nCost Function Over Epochs\n");
    printf("%8.4f ┐\n", history->max);
    for (int i = 0; i < GRAPH_HEIGHT; i++) {
        printf("%s\n", graph[i]);
    }
    printf("%8.4f ┴", history->min);
    for (int i = 0; i < GRAPH_WIDTH-10; i++) printf("─");
    printf(" %d epochs\n", history->count);
    
    // Print epoch markers
    printf("        ");  // Align with graph
    for (int i = 0; i <= 4; i++) {
        printf("%-12d", i * history->count / 4);
    }
    printf("\n");
    
    // Save to CSV for external plotting
    FILE *fp = fopen("cost_history.csv", "w");
    if (fp) {
        fprintf(fp, "epoch,cost\n");
        for (int i = 0; i < history->count; i++) {
            fprintf(fp, "%d,%.6f\n", i, history->values[i]);
        }
        fclose(fp);
        printf("\nCost history saved to 'cost_history.csv'\n");
    }

    fflush(stdout);
}