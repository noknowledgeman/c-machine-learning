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
        fprintf(stderr, "The input is not %ud x %d\n", network->layers[0].weights.cols, 1);
    }
    if (out->rows != network->layers[network->num_layers-1].weights.rows || out->cols != 1) {
        fprintf(stderr, "The output is not %ud x %d\n", network->layers[network->num_layers-1].weights.rows, 1);
    }
    
    for (int i = 0; i < network->num_layers; i++) {
        Layer *current = network->layers + i; 
        
        // not great but avoids memory leak, might lead to false frees 
        matDestroy(&current->x);
        current->x = matCopy(in);
        
        Matrix temp = matCreate(current->biases.rows, 1);
        if (temp.data == NULL) return 1;
        if (matMul(&temp, current->weights, in) != 0) return 1;
        if (matAdd(&temp, temp, current->biases) != 0) return 1;
        
        matDestroy(&current->z);
        current->z = matCopy(temp);
        
        // ReLu ever layer except the last, there is soft max
        if (i < network->num_layers - 1) {
            matReLu(&temp, temp);
        } else {
            matSoftMax(&temp, temp);
        }
        matDestroy(&current->a);
        current->a = matCopy(temp);
        
        // only frees it if it is not owned by the caller
        if (i != 0) matDestroy(&in);
        in = temp;
    }
    
    memcpy(out->data, in.data, out->cols*out->rows*sizeof(float));
    matDestroy(&in);
    return 0;
}

// layers has to have the same length of network->num_layers
static int nnAddGradients(NeuralNetwork *network, Layer *gradients, float learning_rate) {
    for (int i = 0; i < network->num_layers; i++) {
        Layer n_layer = network->layers[i];
        
        Matrix temp = matCopy(gradients[i].weights);
        if (matScale(&temp, temp, learning_rate) != 0) return 1;
        if (matSub(&n_layer.weights, n_layer.weights, temp) != 0) return 1;
        matDestroy(&temp);
        
        temp = matCopy(gradients[i].biases);
        if (matScale(&temp, temp, learning_rate) != 0) return 1;
        if (matSub(&n_layer.biases, n_layer.biases, temp) != 0) return 1;
        matDestroy(&temp);
    }
    
    return 0;
}

static int nnBackward(NeuralNetwork *network, Matrix target, float learning_rate) {
    (void)network;
    
    // output after soft max
    Matrix out = network->layers[network->num_layers-1].a;
    if (target.rows != out.rows && target.cols != out.cols) return 1; 
    
    // focus on the weighs first, the array  of errors
    Matrix *ds = (Matrix *)malloc(sizeof(Matrix)*network->num_layers);
    
    // dL is the gradient of C with respect of the pre activation outputs on layer L, L is the last
    Matrix dL = matCreate(target.rows, target.cols);
    if (matSub(&dL, out, target) != 0) return 1;
    ds[network->num_layers-1] = dL;
    
    for (int l = network->num_layers-2; l >= 0; l++) {
        Layer layer = network->layers[l];
        Layer next_layer = network->layers[l+1];
        
        Matrix templ = matCreate(layer.weights.rows, layer.weights.cols);
        if (templ.data == NULL) {
            printf("Could not create\n");
            return 1;
        }
        
        if (matTranspose(&templ, next_layer.weights) != 0) {
            printf("Could not transpose\n");
            return 1;
        };
        
        if (matMul(&templ, templ, ds[l+1]) != 0) {
            printf("Could not multiply\n");
            return 1;
        }
        
        Matrix tempr = matCopy(layer.z);
        
        if (matReLuDer(&tempr, tempr) != 0) return 1;
        
        if (matProduct(&templ, templ, tempr) != 0) return 1;
        matDestroy(&tempr);
        
        ds[l] = templ;
    }
    
    // Will be directly added
    Layer *gradients = (Layer *)malloc(sizeof(Layer)*network->num_layers);
    
    for (int i = 0; i < network->num_layers; i++) {
        // copying for now
        gradients[i].biases = ds[i];
        
        gradients[i].weights = matCopy(network->layers[i].weights);
        // this should be the weights
        printf("weight size %d x %d\n", network->layers[i].weights.rows, network->layers->weights.cols);
        printf("%d x %d mul %d x %d\n", network->layers[i].x.rows, network->layers[i].x.cols, ds[i].rows, ds[i].cols);
        
        // Idk the math yet
        for (int j = 0; j < network->layers[i].weights.rows; j++) {
            for (int k = 0; k < network->layers[i].weights.cols; k++) {
                gradients[i].weights.data[j*gradients[i].weights.cols + k] = network->layers[i].x.data[k]*ds->data[j];
            }
        }
    }
    
    printf("Found the gradients for the backpropagation!\n");
    
    if (nnAddGradients(network, gradients, learning_rate) != 0) return 1;
    
    // freeing the gradients
    for (int i = 0; i < network->num_layers; i++) {
        matDestroy(&gradients[i].weights);
        matDestroy(&gradients[i].biases);
    }
    free(gradients);
    free(ds);
    
    return 0;
}

static void nnDestroy(NeuralNetwork *network) {
    for (int i = 0; i < network->num_layers; i++) {
        Layer current = network->layers[i];
        matDestroy(&current.biases);
        matDestroy(&current.weights);
        matDestroy(&current.x);
        matDestroy(&current.z);
        matDestroy(&current.a);
    }
}

#endif