
# Machine Learning in C

> **_NOTE:_** this is still a long work in progress

A neural network implementation in C used as a form to learn neural networks, gradients, and backpropagation. This was used as a learning project to teach myself machine learning, especially helpful as preliminary work for my Thesis which is heavily machine learning based.

## Evaluation

At current commit
- After all training images 1 epoch accurcy 86.81%
- After all training images 2 epochs accurcy 87.02%
- After 5 epochs it has decreased to 87.1 from 88, probably because it is not being shuffled

with shuffled indeces:

- Epoch 0: 90.06%, Epoch 1: 90.33%, Epoch 5: 88.91% 

With the Layers 28*28 -> 128 -> 10:

- Epoch 1: 75.84% accurate

With learning rate 0.01
After 1 epoch 97%

With the layers 28*28 -> 256 -> 128 -> 10

After 5 epochs 97.6%

## TODO

- [x] Backpropagation
- [x] Shuffling
- [x] Different sized models
- [x] Plotting accuracies
- [x] Batching (semi)
- [ ] Automatic dataset downloading
- [ ] Momentum
- [ ] Adam
- [ ] Saving a model to disk (Either a straight dump or look into compatible filetypes, like gguf or pth)
- [ ] pthreads
- [ ] Different activation functions
- [ ] Convolutional neural networksk
- [ ] GPU acceleration

## Running

```bash
./datasets.sh mnist
make test
```
> this only works if you have the mnist dataset in a directory called mnist in the right format of idxn-ubyte. See the main.c file for the right file names.

## Resources

A great help to understand Backpropagation was [Neural Networks](http://neuralnetworksanddeeplearning.com/chap2.html)