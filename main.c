#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>

#define MATRIX_IMPLEMENTATION
#include "matrix.h"
#include "types.h"
#include "neuralnetwork.h"

#define IMAGE_SIZE 28*28


// row_major
typedef struct {
    u8 data[IMAGE_SIZE];
} Image;

typedef struct {
    Image image;
} LabelledImage;

// easier for now
typedef struct {
    Image *images;
    u8 *labels;
    u32 num_images;
} LabelledImages;

u32 swap32(u32 x) {
    return ((x >> 24) & 0x000000FF) |
           ((x >> 8)  & 0x0000FF00) |
           ((x << 8)  & 0x00FF0000) |
           ((x << 24) & 0xFF000000);
}

// the length of the image and the label file should be the same
// You own the array
// returns sucess == 1
int decodeImages(LabelledImages *labelled_images, const char *images, const char *labels) {
    // read images
    FILE *image_file = fopen(images, "r");
    
    if (fgetc(image_file) != 0) return 0;
    if (fgetc(image_file) != 0) return 0;
    
    u8 type = fgetc(image_file);
    if (type != 0x08) return 0;
    u8 num_dims = fgetc(image_file);
    if (num_dims != 0x03) return 0;
    // printf("type: %d, dimensions: %d\n", type, num_dims);
    
    u32 dimensions[3] = {0};
    fread(dimensions, sizeof(int), 3, image_file);
    for (int i =0; i < 3; i++) {
        dimensions[i] = swap32(dimensions[i]);
    }
    
    // printf("dimensions: (%d, %d, %d)\n", dimensions[0], dimensions[1], dimensions[2]);
    
    labelled_images->images = malloc(dimensions[0]*dimensions[1]*dimensions[2]);
    if (labelled_images->images == NULL) return 0;
    fread(labelled_images->images, 1, dimensions[0]*dimensions[1]*dimensions[2], image_file);
    fclose(image_file);
    
    // read images
    FILE *label_file = fopen(labels, "r");
    
    if (fgetc(label_file) != 0) return 0;
    if (fgetc(label_file) != 0) return 0;
    
    type = fgetc(label_file);
    if (type != 0x08) return 0;
    num_dims = fgetc(label_file);
    if (num_dims != 0x01) return 0;
    // printf("Labels type: %d, dimensions: %d\n", type, num_dims);
    
    u32 label_dimension;
    fread(&label_dimension, sizeof(int), 1, label_file);
    label_dimension = swap32(label_dimension);
    
    // printf("label_dimension: %d\n", label_dimension);
    
    labelled_images->labels = malloc(label_dimension);
    fread(labelled_images->labels, 1, label_dimension, label_file);
    fclose(label_file);
    
    labelled_images->num_images = label_dimension;
    
    return 1;
}

void freeLabelledImages(LabelledImages images) {
    free(images.images);
    free(images.labels);
}
 
void showImage(Image *image) {
    const char options[9] = " .:+=x$#";
    for (int i = 0; i < 28; i++) {
        printf("=");
    }
    printf("\n");
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            u8 idx = (image->data[28*i + j])/32;
            printf("%c", options[idx]);
        }
        printf("\n");
    }
    printf("\n");
}

// -------------------------------------------------------- Machine Learnign

// Cross entropy
float loss(Matrix learned, int real) {
    if (learned.cols != 1 && learned.rows != 10) {
        return -1.0f;
    }
    
    return -logf(learned.data[real]);
}

int main() {
    LabelledImages training_images;
    if (!decodeImages(&training_images, "./mnist/train-images.idx3-ubyte", "./mnist/train-labels.idx1-ubyte")) {
        printf("Error when reading the file");
        return 1;
    }
    LabelledImages test_images;
    if (!decodeImages(&test_images, "./mnist/t10k-images.idx3-ubyte", "./mnist/t10k-labels.idx1-ubyte")) {
        printf("Error when reading the file");
        return 1;
    }
    
    // showing a random image
    // srand(time(NULL));
    // u32 idx = (rand() % training_images.num_images);
    // Image *image = training_images.images + idx;
    // printf("label: %d\n", training_images.labels[idx]);
    // showImage(image);
    
    // implementing an MLP]
    // so there will be 3 layers 28*28 -> 256 -> 128 -> 10
    
    NeuralNetwork network = {0};
    // just a one layer network with output 10
    nnCreate(&network, 28*28, 1, 10);
    
    // randomm init
    Layer layer = network.layers[0];
    for (int i = 0; i < layer.weights.cols*layer.weights.rows; i++) {
        layer.weights.data[i] = (float)rand()/(float)RAND_MAX;
    }
    
    
    Matrix in = matCreate(IMAGE_SIZE, 1);
    Matrix out = matCreate(10, 1);
    for (int i = 0; i < 5; i++) {
        // choosing the image
        Image *image = training_images.images + i;
        u32 label = training_images.labels[i];
        printf("Label: %ud\n", label);
        showImage(image);
        
        // flattening the image
        for (int j = 0; j < IMAGE_SIZE; j++) {
            in.data[i] = (float)image->data[i]/255.0;
        }
        
        matDebug(out);
        assert(nnForward(&network, &out, in) == 0);
        
        matDebug(out);
        
        int max_idx;
        float curr_max = -1.0;
        for (int i = 0; i < 10; i++) {
            if (curr_max < out.data[i]) {
                curr_max = out.data[i];
                max_idx = i;
            }
        }
        printf("Found %d\n", max_idx);
        
        Matrix actual = matCreate(10, 1);
        actual.data[label] = 1.0;
        assert(nnBackward(&network, actual, 0.001) == 0);
        matDestroy(&actual);
    }
    
    
    freeLabelledImages(training_images);
    freeLabelledImages(test_images);
    
    matDestroy(&in);
    matDestroy(&out);
    nnDestroy(&network);
    
    return 0;
}