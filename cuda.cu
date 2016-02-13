#include <stdio.h>
#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "board.h"

#define N (2048 * 2048)
#define THREADS_PER_BLOCK 512
#define BRANCHING_FACTOR 20

__global__ void device(int *input, int *output) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    //Board *board = (Board*) input[index];
    printf("test\n");
}

// main routine that executes on the host
int main(void) {
    unsigned char x = 4;
    unsigned char y = 4;

    int inCount = 1;
    Board *inputBoards = new Board[inCount];
    inputBoards[0] = new Board(x, y);
    int inputSize = inCount * sizeof(Board);

    int outCount = inCount * BRANCHING_FACTOR;
    Board *outputBoards = new Board[outCount];
    int outputSize = outCount * sizeof(Board);

    int *dev_input;
    int *dev_output;

    cudaMalloc((void**) &dev_input, inputSize);
    cudaMalloc((void**) &dev_output, outputSize);

    cudaMemcpy(dev_input, &inputBoards, inputSize, cudaMemcpyHostToDevice);

    cout << "device method starting..." << endl;
    device<<<N / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(dev_input, dev_output);
    cout << "device method end" << endl;

    cudaDeviceSynchronize();

    cudaMemcpy(outputBoards, dev_output, outputSize, cudaMemcpyDeviceToHost);

    cudaFree(dev_input);
    cudaFree(dev_output);
    return 0;
}
