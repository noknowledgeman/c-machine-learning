#ifndef MATRIX_H
#define MATRIX_H

#include "arena.h"
typedef struct Matrix {
    int rows;
    int cols;
    float *data;
} Matrix;

// return 0 on success 1 otherwise
// 0 initialized
Matrix matCreate(unsigned int rows, unsigned int cols);
void matDestroy(Matrix *a);
Matrix matDupe(Matrix a);

// return 0 on success 1 otherwise
// 0 initialized
Matrix matArenaCreate(Arena *arena, unsigned int rows, unsigned int cols);
Matrix matArenaDupe(Arena *arena, Matrix a);

int matMul(Matrix *out, Matrix a, Matrix b);
int matScale(Matrix *out, Matrix a, float r);
int matAdd(Matrix *out, Matrix a, Matrix b);
int matSub(Matrix *out, Matrix a, Matrix b);
int matTranspose(Matrix *out, Matrix a);
// [a, b]prod[d, e] = [a*d, b*e]
int matProduct(Matrix *out, Matrix a, Matrix b);

// not sure how to implement it yet
int matSoftMax(Matrix *out, Matrix a);

int matReLu(Matrix *out, Matrix a);
int matReLuDer(Matrix *out, Matrix a);

void matDebug(Matrix a);

// #define MATRIX_IMPLEMENTATION
#ifdef MATRIX_IMPLEMENTATION

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

// zero-initialized
Matrix matCreate(unsigned int rows, unsigned int cols) {
    float *data = (float *)calloc(rows*cols, sizeof(float));
    
    if (data == NULL) {
        return (Matrix) {0};
    }
    
    return (Matrix){
        .rows = (int)rows, 
        .cols = (int)cols,
        .data = data,
    };
}

void matDestroy(Matrix *a) {
    if (a->data != NULL) free(a->data);
    a->data = NULL;
}

Matrix matDupe(Matrix a) {
    Matrix ret = matCreate(a.rows, a.cols);
    
    memcpy(ret.data, a.data, a.rows*a.cols*sizeof(float));
    return ret;
}

Matrix matArenaCreate(Arena *arena, unsigned int rows, unsigned int cols) {
    float *data = (float *)arenaAlloc(arena, rows*cols*sizeof(float));
    
    if (data == NULL) {
        return (Matrix) {0};
    }
    
    return (Matrix){
        .rows = (int)rows, 
        .cols = (int)cols,
        .data = data,
    };
}

Matrix matArenaDupe(Arena *arena, Matrix a) {
    Matrix ret = matArenaCreate(arena, a.rows, a.cols);
    
    memcpy(ret.data, a.data, a.rows*a.cols*sizeof(float));
    return ret;
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

int matScale(Matrix *out, Matrix a, float b) {
    if (!(out->cols == a.cols && out->rows == a.rows)) return 1;
    
    for (int i = 0; i < a.cols*a.rows; i++) {
        out->data[i] = b*a.data[i];
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

int matSub(Matrix *out, Matrix a, Matrix b) {
    if (!(out->cols == a.cols && out->rows == a.rows)) return 1;
    if (!(b.cols == a.cols && b.rows == a.rows)) return 1;
    for (int i = 0; i < a.rows*a.cols; i++) {
        out->data[i] = a.data[i] - b.data[i];
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

int matProduct(Matrix *out, Matrix a, Matrix b) {
    if (!(out->cols == a.cols && out->rows == a.rows)) return 1;
    if (!(b.cols == a.cols && b.rows == a.rows)) return 1;
    
    for (int i = 0; i < a.rows*a.cols; i++) {
        out->data[i] = a.data[i]*b.data[i];
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

int matReLuDer(Matrix *out, Matrix a) {
    if (!(out->cols == a.cols && out->rows == a.rows)) return 1;
    
    for (int i = 0; i < a.rows*a.cols; i++) {
        out->data[i] = (a.data[i] > 0) ? 1 : 0;
    }
    
    return 0;
}

int matSoftMax(Matrix *out, Matrix a) {
    if (!(out->cols == a.cols && out->rows == a.rows)) return 1;

    float max_val = a.data[0];
    for (int i = 1; i < a.rows*a.cols; i++) {
        if (a.data[i] > max_val) max_val = a.data[i];
    }

    float total = 0.0;
    for (int i = 0; i < a.rows*a.cols; i++) {
        float curr = exp(a.data[i] - max_val);
        out->data[i] = curr;
        total += curr;
    }

    for (int i = 0; i < a.rows*a.cols; i++) {
        out->data[i] /= total;
    }

    return 0;
}

void matDebug(Matrix mat) {
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            printf("%5.1f ", mat.data[i*mat.cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

#endif
#endif