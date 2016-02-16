#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>

#define CUDA_ERROR_CHECK
#define CUDA_SAFE_CALL( err ) __CUDA_SAFE_CALL( err, __FILE__, __LINE__ )
#define CUDA_CHECK_ERROR()    __CUDA_CHECK_ERROR( __FILE__, __LINE__ )
#define GET_INDEX(y_max, x, y) (y_max * x + y)

using namespace std;

typedef struct {
    uint64_t bitboards[2];
} Board;

void outOfMemHandler() {
    std::cerr << "Unable to satisfy request for memory\n";
    std::abort();
}

inline void __CUDA_SAFE_CALL( cudaError err, const char *file, const int line ) {
    #ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err ) {
        fprintf( stderr, "CUDA_SAFE_CALL() failed at %s:%i : %s\n", file, 
                line, cudaGetErrorString( err ) );
        exit( -1 );
    }
    #endif
}

inline void __CUDA_CHECK_ERROR( const char *file, const int line ) {
    #ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err ) {
        fprintf( stderr, "CUDA_CHECK_ERROR() failed at %s:%i : %s\n", file, 
                line, cudaGetErrorString( err ) );
        exit( -1 );
    }
    
    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err ) {
        fprintf( stderr, "CUDA_CHECK_ERROR() with sync failed at %s:%i : %s\n", 
                file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
    #endif
}

inline bool boards_equal(Board *b1, Board *b2) {
    return b1->bitboards[0] == b2->bitboards[0] && b1->bitboards[1] == b2->bitboards[1];
}

int compare_boards(const void *v1, const void *v2) {
    Board *b1 = (Board*) v1;
    Board *b2 = (Board*) v2;
    return b1->bitboards[0] < b2->bitboards[0];
}

__host__ __device__ inline bool get_location(Board *board, int y_max, int x, int y) {
    //TODO this will probs break for bigger boards
    int index = GET_INDEX(y_max, x, y);
    int boardIndex;
    double offset;
    
    if (index < 63) { // board 1
        boardIndex = 0;
        offset = pow(2.0, index + 1);
    } else { // board 2
        boardIndex = 1;
        offset = pow(2.0, index - 63);
    } 
    return (board->bitboards[boardIndex] & (int) offset) != 0;
}

__device__ inline void set_location(Board *board, int y_max, int x, int y) {
    //TODO this will probs break for bigger boards
    int index = GET_INDEX(y_max, x, y);
    int boardIndex;
    int offset;

    if (index < 63) { // board 1
        boardIndex = 0;
        offset = pow(2.0, index + 1);
    } else { // board 2
        boardIndex = 1;
        offset = pow(2.0, index - 63);
    } 
    board->bitboards[boardIndex] = board->bitboards[boardIndex] | offset;
}


__host__ __device__ inline bool is_valid(Board *board) {
    return board->bitboards[0] & 1 != 0;
}

__host__ __device__ inline void set_valid(Board *board, bool valid) {
    if (valid) {
        board->bitboards[0] = board->bitboards[0] | 1;    
    } else {
        board->bitboards[0] = board->bitboards[0] & 0;
    }
}

__host__ __device__ void print_board(Board *board, int x_max, int y_max) {
    printf("size: (%d, %d)\n", x_max, y_max);
    printf("   ");
    for (int y = 0 ; y < y_max ; y++) {
        printf("%d  ", y);
    }
    printf("\n");
    for (int x = 0 ; x < x_max ; x++) {
        printf("%d ", x);
        for (int y = 0 ; y < y_max ; y++) {
            if (get_location(board, y_max, x, y) == true) {
                printf(" X ");
            } else {
                printf(" . ");
            }
        }
        printf("\n");
    }
}

// blockIdx.x  = block index within the grid
// blockDim.x  = dimension of the block
// threadIdx.x = thread index within the block

__global__ void next_boards(Board *input, Board *output, int branching, 
        int x_max, int y_max, bool vertical) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    Board board = input[index];
    int count = 0; 
    if (is_valid(&board)) { 
        if (vertical) {
            for (int x = 0 ; x < x_max - 1; x++) {
                for (int y = 0 ; y < y_max ; y++) {
                    if (!get_location(&board, y_max, x, y) && 
                            !get_location(&board, y_max, x + 1, y)) {
                        memcpy(&output[index * branching + count], &board, sizeof(Board));
                        set_location(&output[index * branching + count], y_max, x, y);
                        set_location(&output[index * branching + count], y_max, x + 1, y); 
                        count++;
                    }
                }
            }
        } else {
            for (int x = 0 ; x < x_max ; x++) {
                for (int y = 0 ; y < y_max - 1 ; y++) {
                    if (!get_location(&board, y_max, x, y) && 
                            !get_location(&board, y_max, x, y + 1)) {
                        memcpy(&output[index * branching + count], &board, sizeof(Board));
                        set_location(&output[index * branching + count], y_max, x, y);
                        set_location(&output[index * branching + count], y_max, x, y + 1);
                        count++;
                    }
                }
            }
        }
    }
    for (int i = 0 ; i < count ; i++) {
        set_valid(&output[index * branching + i], true);
    }
    for (int i = count ; i < branching ; i++) {
        set_valid(&output[index * branching + i], false);
    }
}

void work_down(Board* input, int x_max, int y_max, int inCount, bool vertical, int depth) {
    if (inCount == 0) {
        printf("\nno more moves at depth: %d\n", depth);
        return;
    }
    printf("\nstarting for for depth: %d\n", depth);
    Board *dev_input;
    Board *dev_output;

    int inputSize = inCount * sizeof(Board);

    //TODO might be wrong branching count
    int branching = x_max * y_max - 2 * depth;
    int outCount = inCount * branching;
    int outputSize = outCount * sizeof(Board);

    printf("input count     : %d\n", inCount);
    printf("branching count : %d\n", branching);
    printf("output count    : %d\n", outCount);

    Board *output = new Board[outCount];

    CUDA_SAFE_CALL(cudaMalloc((void **) &dev_input, inputSize));
    CUDA_SAFE_CALL(cudaMalloc((void **) &dev_output, outputSize));
    
    CUDA_SAFE_CALL(cudaMemcpy(dev_input, input, inputSize, cudaMemcpyHostToDevice));
    
    next_boards<<<inCount, 1>>>(dev_input, dev_output, branching, x_max, y_max, vertical);
    CUDA_CHECK_ERROR();
    
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaMemcpy(output, dev_output, outputSize, cudaMemcpyDeviceToHost));

    bool next = false;
    for (int i = 0 ; i < outCount ; i++ ) {
        Board board = (Board) output[i];
        if (is_valid(&board) == true) {
            next = true;
            break;
        }
    }
    
    CUDA_SAFE_CALL(cudaFree(dev_input));
    CUDA_SAFE_CALL(cudaFree(dev_output));
    
    if (next) {
        int validCount = 0;
        Board *validOutput = new Board[outCount];
        for (int i = 0 ; i < outCount ; i++) {
            if (is_valid(&output[i]) == true) {
                memcpy(&validOutput[validCount++], &output[i], sizeof(Board));
            }
        }
        
        // sorts the new output so that duplicates can be removed
        qsort(validOutput, validCount, sizeof(Board), compare_boards);
        Board *noDuplicates = new Board[outCount];
        int dupCount = 1;

        Board last = validOutput[0];
        memcpy(&noDuplicates[0], &validOutput[0], sizeof(Board));
        
        for (int i = 1 ; i < validCount ; i++) {
            if (!boards_equal(&last, &validOutput[i])) {
                memcpy(&noDuplicates[dupCount++], &validOutput[i], sizeof(Board));
                last = validOutput[i];
            }
        }
        printf("valid count     : %d\n", validCount);
        printf("duplicate count : %d\n", dupCount);
        delete[] output;
        delete[] validOutput;
        work_down(noDuplicates, x_max, y_max, dupCount, !vertical, depth + 1);
        delete[] noDuplicates;
    } else {
        printf("no more moves\n");
        delete[] output;
    }
}

// main routine that executes on the host
int main(void) {
    unsigned char x = 30;
    unsigned char y = 2;

    std::set_new_handler(outOfMemHandler);

    printf("Board size: %d\n", sizeof(Board));

    int inCount = 1;
    Board *inputBoards = new Board[inCount];
    inputBoards[0].bitboards[0] = 0;
    inputBoards[0].bitboards[1] = 0;
    set_valid(&inputBoards[0], true);
    printf("initial\n");
    print_board(&inputBoards[0], x, y);
    work_down(inputBoards, x, y, 1, true, 0);
    delete[] inputBoards; 
    return 0;
}
