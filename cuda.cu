#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>

#define N (2048 * 2048)
#define THREADS_PER_BLOCK 512

using namespace std;

typedef struct {
    bool valid;
    unsigned char x_max;
    unsigned char y_max;
    bool player;
    uint64_t bitboards[2];
} Board;

void generate_board(Board *board, unsigned char x, unsigned char y) {
    board->valid = true;
    board->x_max = x;
    board->y_max = y;
    board->player = true;
    //int num = ceil((1.0 * x * y) / (8 * sizeof(uint64_t)));
    int num = 2;
    for (int i = 0 ; i < num ; i++) {
        board->bitboards[i] = 0;
    }
}

__host__ __device__ inline int get_index(Board *board, int x, int y) {
    return ((int) board->y_max) * x + y;
}

__host__ __device__ inline uint64_t get_offset(int index, int boardIndex) {
    int bitNum = index - (boardIndex * 8 * sizeof(uint64_t));
    return pow ((double) 2, bitNum);
}

__host__ __device__ inline bool get_location(Board *board, int x, int y) {
    int index = get_index(board, x, y);
    int boardIndex = index / (8 * sizeof(uint64_t));
    uint64_t offset = get_offset(index, boardIndex);
    return (board->bitboards[boardIndex] & offset) != 0;

}

__device__ inline void set_location(Board *board, int index) {
    int boardIndex = index / (8 * sizeof(uint64_t));
    uint64_t offset = get_offset(index, boardIndex);
    board->bitboards[boardIndex] = board->bitboards[boardIndex] | offset;
}

__host__ __device__ void board_print(Board *board) {
    printf("size: (%d, %d)\n", board->x_max, board->y_max);
    printf("   ");
    for (int y = 0 ; y < board->y_max ; y++) {
        printf("%d  ", y);
    }
    printf("\n");
    for (int x = 0 ; x < board->x_max ; x++) {
        printf("%d ", x);
        for (int y = 0 ; y < board->y_max ; y++) {
            if (get_location(board, x, y) == true) {
                printf(" X ");
            } else {
                printf(" . ");
            }
        }
        printf("\n");
    }
}

__device__ void copy_board(Board *dest, Board *src) {
    dest->valid = true;
    dest->x_max = src->x_max;
    dest->y_max = src->y_max;
    dest->player = src->player;
    //int num = ceil((1.0 * dest->x_max * dest->y_max) / (8 * sizeof(uint64_t)));
    int num = 2;
    //dest->bitboards = new uint64_t[num];
    memcpy(dest->bitboards, src->bitboards, num * sizeof(uint64_t));
}

// blockIdx.x  = block index within the grid
// blockDim.x  = dimension of the block
// threadIdx.x = thread index within the block

/**
 * Calculates the next boards on the GPU.
 * input: pointer to the array of boards.
 * output: pointer of where to write out the next moves;
 * branching: the branching factor for the move. It is used to
 *      calulcate what index to write the output
 */
__global__ void next_boards(Board *input, Board *output, int branching) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    printf("index: %d, branching: %d\n", index, branching);
    Board board = input[index];
    int count = 0; 
    
    for (int x = 0 ; x < board.x_max - 1; x++) {
        for (int y = 0 ; y < board.y_max ; y++) {
            if (!get_location(&board, x, y) && !get_location(&board, x + 1, y)) {
                printf("found spot at (%d, %d): %d\n", x, y, count);
                // move found for (x, y) (x + 1, y)
                Board newBoard = output[index + count];
                count++;
                copy_board(&newBoard, &board);
                set_location(&newBoard, get_index(&newBoard, x, y));
                set_location(&newBoard, get_index(&newBoard, x + 1, y));
                board_print(&newBoard);
            }
        }
    }

    for (int i = count ; i < branching ; i++) {
        printf("filling in space: %d\n", i);
        output[index + i].valid = false;
    }
}

// main routine that executes on the host
int main(void) {
    unsigned char x = 2;
    unsigned char y = 2;
    int branching = x * y;

    int inCount = 1;
    Board *inputBoards = new Board[inCount];
    generate_board(&inputBoards[0], x, y);
    int inputSize = inCount * sizeof(Board);

    int outCount = inCount * branching;
    Board *outputBoards = new Board[outCount];
    int outputSize = outCount * sizeof(Board);

    board_print(&inputBoards[0]);

    Board *dev_input;
    Board *dev_output;

    cout << "mallocing..." << endl;
    cudaMalloc((void**) &dev_input, inputSize);
    cudaMalloc((void**) &dev_output, outputSize);

    cudaMemcpy(dev_input, inputBoards, inputSize, cudaMemcpyHostToDevice);

    cout << "device method starting..." << endl;
    //device<<<N / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(dev_input, dev_output);
    next_boards<<<1, inCount>>>(dev_input, dev_output, branching);
    cout << "device method end" << endl;

    cudaDeviceSynchronize();
    cudaMemcpy(outputBoards, dev_output, outputSize, cudaMemcpyDeviceToHost);

    for (int i = 0 ; i < outCount ; i++ ) {
        Board board = (Board) outputBoards[i];
        if (board.valid == true) {
            printf("--------------------------\n");
            board_print(&board);
        } else {
            cout << board.valid << endl;
            printf("not valid\n");
        }
    }

    cudaFree(dev_input);
    cudaFree(dev_output);
    return 0;
}
