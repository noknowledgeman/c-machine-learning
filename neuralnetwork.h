// functions are static for now
#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "matrix.h"
#include "types.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
 
typedef struct {
    // layer size x previous layer size
    Matrix weights;
    // size of layer x 1
    Matrix biases;
    
    // for backpropagation maybe add x z and a
} Layer;

typedef struct {
    // the first layer is first hidden layer the layer num_layers-1 is the output 
    // currently activation functionn ReLu is assumed between the layers
    // Last layer has no biases
    // The layers have to match, the programm will check this
    // Max 5 layers for now to avoid heap allocation
    Layer layers[5];
    int num_layers;
} NeuralNetwork;

// at least two variadics, for now only ReLu and nothing at the end
// num_layers >= 1;
static int nnCreate(NeuralNetwork *network, u32 in_size, u32 num_layers, ...) {
    network->num_layers = num_layers;
    if(num_layers > 5) return 1;
    
    va_list args;
    va_start(args, num_layers);
    
    for (int i = 0; i < num_layers; i++) {
        u32 size = va_arg(args, u32);
        Matrix weights = matCreate(size, in_size);
        
        network->layers[i].weights = weights;
        
        Matrix biases = matCreate(size, 1);
        if (biases.data == NULL) return 1;
        network->layers[i].biases = biases;
    }
    
    va_end(args);
    
    return 0;
}


// in has to be the size of network->layers[0].weights.cols x 1 and out has to be 
// the same size of network->weights[network->num_weights-a] x 1
// network is by reference but is constant
static int nnForward(NeuralNetwork *network, Matrix *out, Matrix in) {
    if (in.rows != network->layers[0].weights.cols || in.cols != 1) {
        fprintf(stderr, "The input is not %ud x %d\n", network->layers[0].weights.cols, 1);
    }
    if (out->rows != network->layers[network->num_layers-1].weights.rows || out->cols != 1) {
        fprintf(stderr, "The output is not %ud x %d\n", network->layers[network->num_layers-1].weights.rows, 1);
    }
    
    for (int i = 0; i < network->num_layers; i++) {
        Layer current = network->layers[i]; 
        
        Matrix temp = matCreate(current.biases.rows, 1);
        if (temp.data == NULL) return 1;
        if (matMul(&temp, current.weights, in) != 0) return 1;
        if (matAdd(&temp, temp, current.biases) != 0) return 1;
        
        // only frees it if it is not owned by the caller
        if (i != 0) matDestroy(in);
        in = temp;
    }
    
    memcpy(out->data, in.data, out->cols*out->rows*sizeof(float));
    matDestroy(in);
    return 0;
}

static int nnBackward(NeuralNetwork *network) {
    (void)network;
    return 0;
}

static void nnDestroy(NeuralNetwork *network) {
    for (int i = 0; i < network->num_layers; i++) {
        Layer current = network->layers[i];
        matDestroy(current.biases);
        matDestroy(current.weights);
    }
}

#endif