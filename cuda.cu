#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <sys/time.h>
#include <iostream>

#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/unique.h>


#define THREADS_PER_BLOCK 1024

#define CUDA_ERROR_CHECK
#define CUDA_SAFE_CALL( err ) __CUDA_SAFE_CALL( err, __FILE__, __LINE__ )
#define CUDA_CHECK_ERROR()    __CUDA_CHECK_ERROR( __FILE__, __LINE__ )
#define GET_INDEX(y_max, x, y) (y_max * x + y)
#define LOG_PRINTF(...) do { \
    struct timeval time_now; \
    gettimeofday(&time_now, NULL); \
    tm* time_str_tm = gmtime(&time_now.tv_sec); \
    printf("%02i:%02i:%02i:%06i : ", time_str_tm->tm_hour, time_str_tm->tm_min, \
            time_str_tm->tm_sec, time_now.tv_usec); \
    printf(__VA_ARGS__); \
} while (0) 
#define LOG_FPRINTF(f, ...) do { \
    struct timeval time_now; \
    gettimeofday(&time_now, NULL); \
    tm* time_str_tm = gmtime(&time_now.tv_sec); \
    fprintf(f, "%02i:%02i:%02i:%06i : ", time_str_tm->tm_hour, time_str_tm->tm_min, \
            time_str_tm->tm_sec, time_now.tv_usec); \
    fprintf(f, __VA_ARGS__); \
} while (0) 

using namespace std;

typedef struct {
    uint64_t bitboards[2];
} Board;

void outOfMemHandler() {
    LOG_FPRINTF(stderr, "Unable to satisfy request for memory\n");
    std::abort();
}

inline void __CUDA_SAFE_CALL( cudaError err, const char *file, const int line ) {
    #ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err ) {
        LOG_FPRINTF(stderr, "CUDA_SAFE_CALL() failed at %s:%i : %s\n", file, 
                line, cudaGetErrorString( err ) );
        exit( -1 );
    }
    #endif
}

inline void __CUDA_CHECK_ERROR( const char *file, const int line ) {
    #ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err ) {
        LOG_FPRINTF(stderr, "CUDA_CHECK_ERROR() failed at %s:%i : %s\n", file, 
                line, cudaGetErrorString(err));
        exit( -1 );
    }
    
    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err ) {
        LOG_FPRINTF( stderr, "CUDA_CHECK_ERROR() with sync failed at %s:%i : %s\n", 
                file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
    #endif
}
__host__ __device__ inline bool is_valid(const Board *board) {
    return board->bitboards[0] & 1 != 0;
}

__host__ __device__ inline void set_valid(Board *board, bool valid) {
    if (valid) {
        board->bitboards[0] = board->bitboards[0] | 1;    
    } else {
        board->bitboards[0] = board->bitboards[0] & 0;
    }
}

inline bool boards_equal(const Board *b1, const Board *b2) {
    return memcmp(b1, b2, sizeof(Board)) == 0;
}





int compare_boards(const void *v1, const void *v2) {
    Board *b1 = (Board*) v1;
    Board *b2 = (Board*) v2;

    if (!is_valid(b1) && !is_valid(b2)) {
        return 0;
    } else if (!is_valid(b1) && is_valid(b2)) {
        return 1;
    } else if (is_valid(b1) && !is_valid(b2)) {
        return -1;
    } else {
        return memcmp(b1, b2, sizeof(Board));
    } 
}

__device__ bool operator ==(const Board& b1, const Board& b2) {
    return b1.bitboards[0] == b2.bitboards[0] && b2.bitboards[1] == b2.bitboards[1];
}

__device__ bool operator <(const Board& b1, const Board& b2) {
    if (!is_valid(&b1) && !is_valid(&b2)) {
        return 0;
    } else if (!is_valid(&b1) && is_valid(&b2)) {
        return -1;
    } else if (is_valid(&b1) && !is_valid(&b2)) {
        return 1;
    } else if (b1.bitboards[0] == b2.bitboards[0]) {
        return b1.bitboards[1] < b2.bitboards[1];
    } else {
        return b1.bitboards[0] < b2.bitboards[0];
    }
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
    return (board->bitboards[boardIndex] & (uint64_t) offset) != 0;
}

__host__ __device__ inline void set_location(Board *board, int y_max, int x, int y) {
    //TODO this will probs break for bigger boards
    int index = GET_INDEX(y_max, x, y);
    int boardIndex;
    uint64_t offset;

    if (index < 63) { // board 1
        boardIndex = 0;
        offset = pow(2.0, index + 1);
    } else { // board 2
        boardIndex = 1;
        offset = pow(2.0, index - 63);
    } 
    board->bitboards[boardIndex] = board->bitboards[boardIndex] | offset;
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

struct is_valid_struct {
    __host__ __device__ bool operator()(const Board b) {
        return !is_valid(&b);
    }
};


__device__ inline void copy_left(Board *board, int x_max, int y_max) {
    int middle = y_max / 2;
    int leftCount = 0;
    int rightCount = 0;
    if (y_max % 2 == 0) {
        for (int x = 0 ; x < x_max ; x++) {
            for (int y = 0 ; y < middle ; y++) {
                if (get_location(board, y_max, x, y) == true) {
                    leftCount++;
                }
            }

            for (int y = middle ; y < y_max ; y++) {
                if (get_location(board, y_max, x, y) == true) {
                    rightCount++;
                }
            }
        }
    } else {
        for (int x = 0 ; x < x_max ; x++) {
            for (int y = 0 ; y < middle ; y++) {
                if (get_location(board, y_max, x, y) == true) {
                    leftCount++;
                }
            }

            for (int y = middle + 1 ; y < y_max ; y++) {
                if (get_location(board, y_max, x, y) == true) {
                    rightCount++;
                }
            }
        }
    }
    if (leftCount < rightCount) {
        Board *tmpBoard = new Board; 
        memcpy(tmpBoard, board, sizeof(Board));
        memset(board, 0, sizeof(Board));
        set_valid(board, true);
        for (int x = 0 ; x < x_max ; x++) {
            for (int y = 0 ; y < y_max ; y++) {
                if (get_location(tmpBoard, y_max, x, y) == true) {
                    set_location(board, y_max, x, y_max - y - 1);
                } 
            }
        }
        delete tmpBoard;
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
        Board *board = &output[index * branching + i];
        set_valid(board, true);
        copy_left(board, x_max, y_max);
    }
    for (int i = count ; i < branching ; i++) {
        set_valid(&output[index * branching + i], false);
    }

    //__syncthreads();
}

int best = 0;

void work_down(Board* input, int x_max, int y_max, int inCount, bool vertical, int depth) {
    if (depth > best) {
        best = depth;
    }
    LOG_PRINTF("\n");
    if (inCount == 0) {
        for (int i = 0 ; i < depth ; i++)
            LOG_PRINTF(" ");
        LOG_PRINTF("no more moves at depth: %d\n", depth);
        return;
    }
    
    LOG_PRINTF("starting for for depth: %d\n", depth);
    Board *dev_input;
    Board *dev_output;

    int inputSize = inCount * sizeof(Board);

    //TODO might be wrong branching count
    int branching = x_max * y_max;
    int outCount = inCount * branching;
    int outputSize = outCount * sizeof(Board);

    
    LOG_PRINTF("best            : %d\n", best);
    LOG_PRINTF("input count     : %d\n", inCount);
    LOG_PRINTF("branching count : %d\n", branching);
    LOG_PRINTF("output count    : %d\n", outCount);

    Board *output = new Board[outCount];
    
    CUDA_SAFE_CALL(cudaMalloc((void **) &dev_input, inputSize));
    CUDA_SAFE_CALL(cudaMalloc((void **) &dev_output, outputSize));
    CUDA_SAFE_CALL(cudaMemcpy(dev_input, input, inputSize, cudaMemcpyHostToDevice));
    
    int blocks = inCount / THREADS_PER_BLOCK;
    blocks = (blocks == 0) ? 1 : blocks;
    
    next_boards<<<blocks, THREADS_PER_BLOCK>>>(dev_input, dev_output, branching, 
            x_max, y_max, vertical);
    CUDA_CHECK_ERROR();
    
    CUDA_SAFE_CALL(cudaMemcpy(output, dev_output, outputSize, cudaMemcpyDeviceToHost));

    LOG_PRINTF("gpu done...\n");

    bool next = false;
    for (int i = 0 ; i < outCount ; i++ ) {
        Board board = (Board) output[i];
        if (is_valid(&board) == true) {
            next = true;
            break;
        }
    }
    
//    CUDA_SAFE_CALL(cudaFree(dev_input));
//    CUDA_SAFE_CALL(cudaFree(dev_output));
    
    if (next) {
        
        LOG_PRINTF("starting to sort...\n");
        thrust::host_vector<Board> h_vec(outCount);
        for (int i = 0 ; i < outCount ; i++) {
            h_vec[i] = output[i];
        }
        delete[] output;
        LOG_PRINTF("copied to host vector\n");
        // copies the host vector to the device
//       thrust::device_vector<Board> d_vec(h_vec.begin(), h_vec.end());

        thrust::device_ptr<Board> dev_ptr = thrust::device_pointer_cast(dev_output);

        LOG_PRINTF("device vector\n"); 
        // removes invalid boards
        thrust::device_vector<Board>::iterator new_end = thrust::remove_if(
                d_vec.begin(), d_vec.end(), is_valid_struct());
        // erases the invalid boards
        d_vec.erase(new_end, d_vec.end());
        LOG_PRINTF("removed invalid\n"); 
        // sorts the boards so duplicates are next to each other 
        thrust::sort(d_vec.begin(), d_vec.end());
       
        // removes the dupliates next to each other
        new_end = thrust::unique(d_vec.begin(), d_vec.end()); 
        d_vec.erase(new_end, d_vec.end());

        // copies the device vector back to the host
        std::vector<Board> stl_vector(d_vec.size());
        thrust::copy(d_vec.begin(), d_vec.end(), stl_vector.begin());
     
        h_vec.clear();
        h_vec.shrink_to_fit();
        d_vec.clear();
        d_vec.shrink_to_fit();

        LOG_PRINTF("actual size     : %d\n", stl_vector.size()); 

        int size = 1000000;
        int outSize = stl_vector.size();
        if (outSize > size) {
            LOG_PRINTF("splitting...\n");
            for (int i = 0 ; i < outSize ; i += size) {
                if (outSize < i + size) {
                    work_down(&stl_vector[i], x_max, y_max, outSize - i, !vertical, depth + 1);
                } else {
                    work_down(&stl_vector[i], x_max, y_max, size, !vertical, depth + 1);
                }
            }

        } else {
            work_down(&stl_vector[0], x_max, y_max, stl_vector.size(), !vertical, depth + 1);
        }
    } else { 
        LOG_PRINTF("no more moves\n");
        delete[] output;
    }
}


// main routine that executes on the host
int main(void) {
    unsigned char x = 7;
    unsigned char y = 7;

    std::set_new_handler(outOfMemHandler);

    LOG_PRINTF("remember to kill the X session\n");
    LOG_PRINTF("Board size: %d\n", sizeof(Board));

    int inCount = 1;
    Board *inputBoards = new Board[inCount];
    memset(&inputBoards[0], 0, sizeof(Board));
    set_valid(&inputBoards[0], true);
    LOG_PRINTF("initial\n");
    print_board(&inputBoards[0], x, y);
    work_down(inputBoards, x, y, 1, true, 0);
    delete[] inputBoards; 
    return 0;
}
