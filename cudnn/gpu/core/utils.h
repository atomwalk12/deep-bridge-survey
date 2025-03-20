#ifndef UTILS_H
#define UTILS_H

#include <cudnn.h>
#include <stdio.h>
#include <string.h>

#define MAX_HISTORY 2000
#define GRAPH_WIDTH 60
#define GRAPH_HEIGHT 20

typedef struct
{
    float values[MAX_HISTORY];
    int count;
    float min;
    float max;
} CostHistory;

extern bool ENABLE_DEBUG_OUTPUT;

void checkCUDNN(cudnnStatus_t status);
void debugDescriptor(const char *name, cudnnTensorDescriptor_t desc);
void debugFilterDescriptor(cudnnFilterDescriptor_t desc);
void debugTensorValues(const char *label, float *device_ptr, int count);
void checkWeightChanges(const char *label, float *device_weights, int size);
void debugMatrixLayout(const char *label, float *device_ptr,
                       int rows, int cols, int max_rows = 3, int max_cols = 5);
void cost_history_init(CostHistory *history);
void cost_history_add(CostHistory *history, float value);
void plot_cost_ascii(CostHistory *history);

#endif // UTILS_H