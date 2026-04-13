// functions are static for now
#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "matrix.h"
#include "types.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
 
typedef struct {
    // layer size x previous layer size
    Matrix weights;
    // size of layer x 1
    Matrix biases;
    
    // for backpropagation maybe add x z and a
    // x: input, z: weighted sum, a: activation
    // The input to the model would be x of the first layer and the output would be a of the last
    Matrix x, z, a;
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
    
    for (int i = 0; i < (int)num_layers; i++) {
        u32 size = va_arg(args, u32);
        Matrix weights = matCreate(size, in_size);
        if (weights.data == NULL) return 1;
        in_size = size;
        
        network->layers[i].weights = weights;
        
        Matrix biases = matCreate(size, 1);
        if (biases.data == NULL) return 1;
        network->layers[i].biases = biases;
    }
    
    va_end(args);
    
    return 0;
}

static void nnDestroy(NeuralNetwork *network) {
    for (int i = 0; i < network->num_layers; i++) {
        Layer *current = &network->layers[i];
        matDestroy(&current->biases);
        matDestroy(&current->weights);
        matDestroy(&current->x);
        matDestroy(&current->z);
        matDestroy(&current->a);
    }
}

// Allocates zero-initialized weight and bias matrices for use as a gradient accumulator
static int nnZeroGradients(NeuralNetwork *network, NeuralNetwork *gradients) {
    gradients->num_layers = network->num_layers;
    for (int i = 0; i < network->num_layers; i++) {
        matDestroy(&gradients->layers[i].weights);
        gradients->layers[i].weights = matCreate(network->layers[i].weights.rows, network->layers[i].weights.cols);
        if (gradients->layers[i].weights.data == NULL) return 1;
        matDestroy(&gradients->layers[i].biases);
        gradients->layers[i].biases = matCreate(network->layers[i].biases.rows, network->layers[i].biases.cols);
        if (gradients->layers[i].biases.data == NULL) return 1;
    }
    return 0;
}

// Creates a 0 initialized NeuralNetwork struct with only the weights and biases initialized to act as the gradient struct
static int nnEnsureGradientsSize(NeuralNetwork *network, NeuralNetwork *out) {
    if (network->num_layers != out->num_layers) return 1;
    for (int i = 0; i < network->num_layers; i++) {
        if (network->layers[i].weights.rows != out->layers[i].weights.rows || network->layers[i].weights.cols != out->layers[i].weights.cols) return 1;
        if (network->layers[i].biases.rows != out->layers[i].biases.rows || network->layers[i].biases.cols != out->layers[i].biases.cols) return 1;
        
        // out->layers[i].weights = matCreate(network->layers[i].weights.rows, network->layers[i].weights.cols);
        // if (out->layers[i].weights.data == NULL) return 1;
        // out->layers[i].biases = matCreate(network->layers[i].biases.rows, network->layers[i].biases.cols);
        // if (out->layers[i].biases.data == NULL) return 1;
    }
    
    return 0;
}

static int nnScaleGradients(NeuralNetwork *gradients, float scale) {
    for (int i = 0; i < gradients->num_layers; i++) {
        if (matScale(&gradients->layers[i].weights, gradients->layers[i].weights, scale) != 0) return 1;
        if (matScale(&gradients->layers[i].biases, gradients->layers[i].biases, scale) != 0) return 1;
    }
    
    return 0;
}

static int nnAddGradients(NeuralNetwork *gradients, NeuralNetwork *batch_gradients) {
    if (gradients->num_layers != batch_gradients->num_layers) return 1;
    for (int i = 0; i < gradients->num_layers; i++) {
        if(matAdd(&gradients->layers[i].weights, gradients->layers[i].weights, batch_gradients->layers[i].weights) != 0) return 1;
        if(matAdd(&gradients->layers[i].biases, gradients->layers[i].biases, batch_gradients->layers[i].biases) != 0) return 1;
    }
    
    return 0;
}

// in has to be the size of network->layers[0].weights.cols x 1 and out has to be 
// the same size of network->weights[network->num_weights-a] x 1
// network is by reference but is constant
// will update x, z, a for each layer
// 
// Currently it has ReLU in each layer and SoftMax at the end as it is specific to MNIST, possibly different activaiton and loss functions after
// 
// some sort of 
// ```c
// typedef struct {
//   int (*activation)(Matrix *, Matrix);
//   int (*activationDer)(Matrix *, Matrix);
// } Activation;
// ```
static int nnForward(NeuralNetwork *network, Matrix *out, Matrix in) {
    if (in.rows != network->layers[0].weights.cols || in.cols != 1) {
        fprintf(stderr, "The input is not %u x %d\n", network->layers[0].weights.cols, 1);
        return 1;
    }
    if (out->rows != network->layers[network->num_layers-1].weights.rows || out->cols != 1) {
        fprintf(stderr, "The output is not %u x %d\n", network->layers[network->num_layers-1].weights.rows, 1);
        return 1;
    }
    
    for (int i = 0; i < network->num_layers; i++) {
        Layer *current = network->layers + i; 
        
        // not great but avoids memory leak, might lead to false frees 
        matDestroy(&current->x);
        current->x = matDupe(in);
        
        Matrix temp = matCreate(current->biases.rows, 1);
        if (temp.data == NULL) return 1;
        if (matMul(&temp, current->weights, in) != 0) return 1;
        if (matAdd(&temp, temp, current->biases) != 0) return 1;
        
        matDestroy(&current->z);
        current->z = matDupe(temp);
        
        // ReLu ever layer except the last, there is soft max
        if (i < network->num_layers - 1) {
            matReLu(&temp, temp);
        } else {
            matSoftMax(&temp, temp);
        }
        matDestroy(&current->a);
        current->a = matDupe(temp);
        
        // only frees it if it is not owned by the caller
        if (i != 0) matDestroy(&in);
        in = temp;
    }
    
    memcpy(out->data, in.data, out->cols*out->rows*sizeof(float));
    matDestroy(&in);
    return 0;
}

// layers has to have the same length of network->num_layers
static int nnAddGradientsToNetwork(NeuralNetwork *network, NeuralNetwork *gradients, float learning_rate) {
    for (int i = 0; i < network->num_layers; i++) {
        Layer n_layer = network->layers[i];
        
        Matrix temp = matDupe(gradients->layers[i].weights);
        if (matScale(&temp, temp, learning_rate) != 0) return 1;
        if (matSub(&n_layer.weights, n_layer.weights, temp) != 0) return 1;
        matDestroy(&temp);
        
        temp = matDupe(gradients->layers[i].biases);
        if (matScale(&temp, temp, learning_rate) != 0) return 1;
        if (matSub(&n_layer.biases, n_layer.biases, temp) != 0) return 1;
        matDestroy(&temp);
    }
    
    return 0;
}

static int nnBackward(NeuralNetwork *network, NeuralNetwork *gradients, Matrix target) {
    // output after soft max
    Matrix out = network->layers[network->num_layers-1].a;
    if (target.rows != out.rows && target.cols != out.cols) return 1; 
    
    // focus on the weighs first, the array  of errors
    Matrix *ds = (Matrix *)malloc(sizeof(Matrix)*network->num_layers);
    
    // dL is the gradient of C with respect of the pre activation outputs on layer L, L is the last
    Matrix dL = matCreate(target.rows, target.cols);
    if (matSub(&dL, out, target) != 0) return 1;
    ds[network->num_layers-1] = dL;
    
    for (int l = network->num_layers-2; l >= 0; l--) {
        Layer *layer = &network->layers[l];
        Layer *next_layer = &network->layers[l+1];
        
        Matrix wT = matCreate(next_layer->weights.cols, next_layer->weights.rows);
        if (wT.data == NULL) {
            printf("Could not create\n");
            return 1;
        }

        if (matTranspose(&wT, next_layer->weights) != 0) {
            printf("Could not transpose\n");
            matDestroy(&wT);
            return 1;
        };

        Matrix templ = matCreate(wT.rows, ds[l+1].cols);
        if (templ.data == NULL) {
            matDestroy(&wT);
            return 1;
        }
        if (matMul(&templ, wT, ds[l+1]) != 0) {
            printf("Could not multiply\n");
            matDestroy(&wT);
            matDestroy(&templ);
            return 1;
        }
        matDestroy(&wT);
        
        Matrix tempr = matDupe(layer->z);
        
        if (matReLuDer(&tempr, tempr) != 0) return 1;
        
        if (matProduct(&templ, templ, tempr) != 0) return 1;
        matDestroy(&tempr);
        
        ds[l] = templ;
    }
    
    // Will be directly added
    // NeuralNetwork gradients = {0};
    // if (nnEnsureGradientsSize(network, gradients) != 0) return 1;
    if (gradients->num_layers != network->num_layers) {
        fprintf(stderr, "The gradient does not have the same layers: %du %du\n", network->num_layers, gradients->num_layers);
        return 1;
    };
    
    for (int i = 0; i < network->num_layers; i++) {
        // copying for now
        gradients->layers[i].biases = ds[i];
        
        gradients->layers[i].weights = matDupe(network->layers[i].weights);
        // this should be the weights
        // printf("weight size %d x %d\n", network->layers[i].weights.rows, network->layers->weights.cols);
        // printf("%d x %d mul %d x %d\n", network->layers[i].x.rows, network->layers[i].x.cols, ds[i].rows, ds[i].cols);
        
        // Idk the math yet
        for (int j = 0; j < network->layers[i].weights.rows; j++) {
            for (int k = 0; k < network->layers[i].weights.cols; k++) {
                gradients->layers[i].weights.data[j*gradients->layers[i].weights.cols + k] = network->layers[i].x.data[k]*ds[i].data[j];
            }
        }
    }
    
    free(ds);
    
    return 0;
}


#endif