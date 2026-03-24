#ifndef MATRIX_H
#define MATRIX_H

typedef struct Matrix {
    int rows;
    int cols;
    float *data;
} Matrix;

// return 0 on success 1 otherwise
// 0 initialized
int matCreate(Matrix *out, unsigned int rows, unsigned int cols);
void matDestroy(Matrix a);

int matMul(Matrix *out, Matrix a, Matrix b);
int matAdd(Matrix *out, Matrix a, Matrix b);
int matReLu(Matrix *out, Matrix a);
int matTranspose(Matrix *out, Matrix a);

// not sure how to implement it yet
int matSoftMax(Matrix *out, Matrix a);

#ifdef MATRIX_IMPLEMENTATION

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int matCreate(Matrix *out, unsigned int rows, unsigned int cols) {
    float *data = (float *)calloc(rows*cols, sizeof(float));
    if (data == NULL) {
        return 1;
    }
    
    *out = (Matrix){
        .rows = rows, 
        .cols = cols,
        .data = data,
    };
    return 0;
}

void matDestroy(Matrix a) {
    free(a.data);
}

// out should be initialized to the right size
int matMul(Matrix *out, Matrix a, Matrix b) {
    if (!(a.cols == b.rows)) {
        fprintf(stderr, "a.cols and b.rows do not match\n");
        return 1;
    };
    if (!(out->rows == a.rows && out->cols == b.cols)) {
        fprintf(stderr, "the out size does not match\n");
        return 1;
    };

    
    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < b.cols; j++) {
            out->data[i*out->cols + j] = 0;
            for (int k = 0; k < a.cols; k++) {
                out->data[i*out->cols + j] += a.data[i*a.cols + k]*b.data[k*b.cols + j];
            }
        }
    }
    
    return 0;
}

int matAdd(Matrix *out, Matrix a, Matrix b) {
    if (!(out->cols == a.cols && out->rows == a.rows)) return 1;
    if (!(b.cols == a.cols && b.rows == a.rows)) return 1;
    
    for (int i = 0; i < a.rows*a.cols; i++) {
        out->data[i] = a.data[i] + b.data[i];
    }
    
    return 0;
}

int matTranspose(Matrix *out, Matrix a) {
    if (!(out->cols == a.rows && out->rows == a.cols)) return 1;
    
    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < a.cols; j++) {
            out->data[j*out->cols + i] = a.data[i*a.cols + j];
        }
    }
    
    return 0;
}

int matReLu(Matrix *out, Matrix a) {
    if (!(out->cols == a.cols && out->rows == a.rows)) return 1;
    
    for (int i = 0; i < a.rows*a.cols; i++) {
        out->data[i] = (a.data[i] > 0) ? a.data[i] : 0;
    }
    return 0;
}

// not sure how to implement it yet
int matSoftMax(Matrix *out, Matrix a) {
    if (!(out->cols == a.cols && out->rows == a.rows)) return 1;
    
    float total = 0.0;
    for (int i = 0; i < (int)a.rows*a.cols; i++) {
        float curr = exp(a.data[i]);
        out->data[i] = curr;
        total += curr;
    }
    
    for (int i = 0; i < (int)a.rows*a.cols; i++) {
        out->data[i] /= total;
    }
    
    return 0;
}

#endif
#endif