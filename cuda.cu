#include "cuda.h"

#define X_MAX 7
#define Y_MAX 7

using namespace std;

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
   #endif
}

__device__ double board_distance(Board *board) {
    double distance = 0;

    // TODO change this
    int middle_x = X_MAX / 2;
    int middle_y = Y_MAX / 2; 

    for (int x = 0 ; x < X_MAX ; x++) {
        for (int y = 0 ; y < Y_MAX ; y++) {
            if (get_location(board, Y_MAX, x, y) == true) {
                distance += sqrt(pow(x - middle_x, 2.0) + pow(y - middle_y, 2.0));
            }
        }
    }
    return distance;
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
        board->distance = board_distance(board);
        //copy_left(board, x_max, y_max);
    }
    for (int i = count ; i < branching ; i++) {
        set_valid(&output[index * branching + i], false);
    }
}

__device__ bool vertical_equal(Board *b1, Board *b2) {
    Board tmp;
    memset(&tmp, 0, sizeof(Board));
    set_valid(&tmp, true);

    for (int x = 0 ; x < X_MAX ; x++) {
        for (int y = 0 ; y < Y_MAX ; y++) {
            if (get_location(b1, Y_MAX, x, y) == true) {
                set_location(&tmp, Y_MAX, x, Y_MAX - y - 1);
            }
        }
    }
    return tmp.bitboards[0] == b2->bitboards[0] && tmp.bitboards[1] == b2->bitboards[1];
}

__device__ bool horizontal_equal(Board *b1, Board *b2) {
    Board tmp;
    memset(&tmp, 0, sizeof(Board));
    set_valid(&tmp, true);

    for (int x = 0 ; x < X_MAX ; x++) {
        for (int y = 0 ; y < Y_MAX ; y++) {
            if (get_location(b1, Y_MAX, x, y) == true) {
                set_location(&tmp, Y_MAX, X_MAX  - x - 1, y);
            }
        }
    }
    return tmp.bitboards[0] == b2->bitboards[0] && tmp.bitboards[1] == b2->bitboards[1];
}


__device__ bool rotate_equal(Board *b1, Board *b2) {
    Board tmp;
    memset(&tmp, 0, sizeof(Board));
    set_valid(&tmp, true);

    for (int x = 0; x < X_MAX; x++) {
        for (int y = 0; y < Y_MAX; y++) {
            if (get_location(b1, Y_MAX, Y_MAX - y - 1, x) == true) {
                set_location(&tmp, Y_MAX, x, y);
            } 
        }
    }
    bool result = tmp.bitboards[0] == b2->bitboards[0] && tmp.bitboards[1] == b2->bitboards[1];
    if (result == true)
        return true;

    memset(&tmp, 0, sizeof(Board));
    set_valid(&tmp, true);

    for (int x = 0 ; x < X_MAX ; x++) {
        for (int y = 0 ; y < Y_MAX ; y++) {
            if (get_location(b1, Y_MAX, x, y) == true) {
                set_location(&tmp, Y_MAX, X_MAX - x - 1, Y_MAX - y - 1);
            }
        }
    }
    result = tmp.bitboards[0] == b2->bitboards[0] && tmp.bitboards[1] == b2->bitboards[1];
    if (result == true)
        return true;

    memset(&tmp, 0, sizeof(Board));
    set_valid(&tmp, true);

    for (int x = 0 ; x < X_MAX ; x++) {
        for (int y= 0 ; y < Y_MAX ; y++) {
            if (get_location(b1, Y_MAX, y, X_MAX - x -1) == true) {
                set_location(&tmp, Y_MAX, x, y);
            }
        }
    }

    return (tmp.bitboards[0] == b2->bitboards[0] && 
            tmp.bitboards[1] == b2->bitboards[1]) ||
        vertical_equal(&tmp, b2) || horizontal_equal(&tmp, b2);
}

__device__ bool operator ==(const Board& b1, const Board& b2) {
    Board board1 = b1;
    Board board2 = b2;
    return (b1.bitboards[0] == b2.bitboards[0] && b2.bitboards[1] == b2.bitboards[1]) ||
        rotate_equal(&board1, &board2) || vertical_equal(&board1, &board2) ||
        horizontal_equal(&board1, &board2);
}

__device__ bool operator <(const Board& b1, const Board& b2) {
    if (!is_valid(&b1) && !is_valid(&b2)) {
        return 0;
    } else if (!is_valid(&b1) && is_valid(&b2)) {
        return -1;
    } else if (is_valid(&b1) && !is_valid(&b2)) {
        return 1;
    } else {
        return b1.distance < b2.distance;
    }
}


void work_down(Board* dev_input, int x_max, int y_max, int inCount, bool vertical, int depth) {
    if (inCount == 0) {
        LOG_PRINTF("no more moves at depth: %d\n", depth);
        return;
    }
    
    int branching = x_max * y_max;
    int outCount = inCount * branching;
    int outputSize = outCount * sizeof(Board);
    Board *dev_output;
    CUDA_SAFE_CALL(cudaMalloc((void **) &dev_output, outputSize));

    printf("\n");
    LOG_PRINTF("depth           : %d\n", depth);
    LOG_PRINTF("input count     : %d\n", inCount);
    LOG_PRINTF("branching count : %d\n", branching);
    LOG_PRINTF("output count    : %d\n", outCount);
    
    int blocks = inCount / THREADS_PER_BLOCK;
    blocks = (blocks == 0) ? 1 : blocks;
    next_boards<<<inCount, 1>>>(dev_input, dev_output, branching, x_max, y_max, vertical);
    CUDA_CHECK_ERROR();

    size_t N = outCount;
    CUDA_SAFE_CALL(cudaFree(dev_input));
    thrust::device_ptr<Board> dev_ptr = thrust::device_pointer_cast(dev_output);
    thrust::device_vector<Board> d_vec(dev_ptr, dev_ptr + N);
    
    // removes
    thrust::device_vector<Board>::iterator new_end = 
        thrust::remove_if(d_vec.begin(), d_vec.end(), is_valid_struct());
    LOG_PRINTF("removed invalid\n");   
    // erases the invalid boards
    d_vec.erase(new_end, d_vec.end());
    
    // sorts the boards so duplicates are next to each other 
    thrust::sort(d_vec.begin(), d_vec.end());
    LOG_PRINTF("sorted\n");
    // removes the dupliates next to each other
    new_end = thrust::unique(d_vec.begin(), d_vec.end()); 
    d_vec.erase(new_end, d_vec.end());

    LOG_PRINTF("removed duplicates\n");

    size_t size = d_vec.size();
    LOG_PRINTF("output size     : %d\n", size);
    const size_t MAX_SIZE = 3000000;
     
    if (size > MAX_SIZE) {
        LOG_PRINTF("splitting...\n");
        for (int i = 0 ; i < size ; i += MAX_SIZE) {
            if (size < i + MAX_SIZE) {
                work_down(&dev_output[i], x_max, y_max, size - i, !vertical, depth + 1);
            } else {
                work_down(&dev_output[i], x_max, y_max, MAX_SIZE, !vertical, depth + 1);
            } 
        }
    } else {
        return work_down(dev_output, x_max, y_max, size, !vertical, depth + 1);
    }
}


// main routine that executes on the host
int main(void) {
    unsigned char x = X_MAX;
    unsigned char y = Y_MAX;

    std::set_new_handler(outOfMemHandler);

    LOG_PRINTF("remember to kill the X session\n");
    LOG_PRINTF("Board size: %d\n", sizeof(Board));

    int inCount = 1;
    size_t inputSize = inCount * sizeof(Board); 
    Board *inputBoards = new Board[inCount];
    memset(&inputBoards[0], 0, sizeof(Board));
    set_valid(&inputBoards[0], true);

    Board* dev_input;
    CUDA_SAFE_CALL(cudaMalloc((void **) &dev_input, inputSize));
    CUDA_SAFE_CALL(cudaMemcpy(dev_input, inputBoards, inputSize, cudaMemcpyHostToDevice));

    LOG_PRINTF("initial\n");
    print_board(&inputBoards[0], x, y);
    work_down(dev_input, x, y, 1, true, 0);
    delete[] inputBoards;
    return 0;
}
