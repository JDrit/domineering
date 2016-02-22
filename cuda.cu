#include "cuda.h"

#define X_MAX 8
#define Y_MAX 8
#define MAX_SIZE 50000

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

    int middle_x = X_MAX / 2;
    int middle_y = Y_MAX / 2; 

    for (int x = 0 ; x < X_MAX ; x++) {
        for (int y = 0 ; y < Y_MAX ; y++) {
            if (get_location(board, Y_MAX, x, y) == true) {
                distance += sqrt(pow(1.0 * x - middle_x, 2.0) + pow(1.0 * y - middle_y, 2.0));
            }
        }
    }
    return distance;
}

// blockIdx.x  = block index within the grid
// blockDim.x  = dimension of the block
// threadIdx.x = thread index within the block

__global__ void next_boards(Board *input, Board *output, double *distances, int branching, 
        bool vertical, int max_index) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int count = 0; 

    // makes sure that the threads only read the given input. This can happen
    // when the amount of threads per block do not line up with the total 
    // input count.
    if (index < max_index) {
        Board board = input[index];
        // only process valid board configurations
        if (is_valid(&board)) { 
            if (vertical) {
                for (int x = 0 ; x < X_MAX - 1; x++) {
                    for (int y = 0 ; y < Y_MAX ; y++) {
                        if (!get_location(&board, Y_MAX, x, y) && 
                                !get_location(&board, Y_MAX, x + 1, y)) {
                            memcpy(&output[index * branching + count], &board, sizeof(Board));
                            set_location(&output[index * branching + count], Y_MAX, x, y);
                            set_location(&output[index * branching + count], Y_MAX, x + 1, y); 
                            count++;
                        }
                    }
                }
            } else {
                for (int x = 0 ; x < X_MAX ; x++) {
                    for (int y = 0 ; y < Y_MAX - 1 ; y++) {
                        if (!get_location(&board, Y_MAX, x, y) && 
                                !get_location(&board, Y_MAX, x, y + 1)) {
                            memcpy(&output[index * branching + count], &board, sizeof(Board));
                            set_location(&output[index * branching + count], Y_MAX, x, y);
                            set_location(&output[index * branching + count], Y_MAX, x, y + 1);
                            count++;
                        }
                    }
                }
            }
        }

        for (int i = 0 ; i < count ; i++) {
            Board *board = &output[index * branching + i];
            set_valid(board, true);
            distances[index * branching + i] = board_distance(board);
        }
        for (int i = count ; i < branching ; i++) {
            set_valid(&output[index * branching + i], false);
            distances[index * branching + i] = -1;
        }
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

int best = 0;

void work_down(Board* input, int inCount, bool vertical, int depth) {
    best = max(best, depth);
    if (inCount == 0) {
        LOG_PRINTF("no more moves at depth: %d\n", depth);
        return;
    }

    int branching = X_MAX * Y_MAX;
    int outCount = inCount * branching;
    
    Board *dev_boards;
    Board *dev_input;
    double *dev_distances;
    
    size_t totalSize = inCount * sizeof(Board) + 
        outCount * sizeof(Board) + 
        outCount * sizeof(double); 
    
    printf("\n");
    LOG_PRINTF("depth           : %d\n", depth);
    LOG_PRINTF("max             : %d\n", best);
    LOG_PRINTF("input count     : %d\n", inCount);
    LOG_PRINTF("branching count : %d\n", branching);
    LOG_PRINTF("output count    : %d\n", outCount);
    LOG_PRINTF("mallocing size  : %zu\n", totalSize);
    
    CUDA_SAFE_CALL(cudaMalloc((void**) &dev_input, inCount * sizeof(Board)));
    CUDA_SAFE_CALL(cudaMemcpy(dev_input, input, inCount * sizeof(Board), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMalloc((void **) &dev_boards, outCount * sizeof(Board)));
    CUDA_SAFE_CALL(cudaMalloc((void **) &dev_distances, outCount * sizeof(double)));

    int blocks = (int) ceil((inCount * 1.0) / THREADS_PER_BLOCK);
    next_boards<<<blocks, THREADS_PER_BLOCK>>>(dev_input, dev_boards, dev_distances, 
            branching, vertical, inCount);
    CUDA_CHECK_ERROR();
    CUDA_SAFE_CALL(cudaFree(dev_input));
    
    size_t N = outCount;
    thrust::device_ptr<Board> d_board_ptr = thrust::device_pointer_cast(dev_boards);
    thrust::device_vector<Board> d_board_vec(d_board_ptr, d_board_ptr + N);

    thrust::device_ptr<double> d_dist_ptr = thrust::device_pointer_cast(dev_distances);
    thrust::device_vector<double> d_dist_vec(d_dist_ptr, d_dist_ptr + N);

    // removes
    thrust::device_vector<Board>::iterator board_end = 
        thrust::remove_if(d_board_vec.begin(), d_board_vec.end(), is_valid_struct());
    thrust::device_vector<double>::iterator dist_end = 
        thrust::remove(d_dist_vec.begin(), d_dist_vec.end(), -1);
    
    // erases the invalid boards
    d_board_vec.erase(board_end, d_board_vec.end());
    d_dist_vec.erase(dist_end, d_dist_vec.end());
    
    // sorts the boards so dupliicates are next to each other 
    // TODO probs not working 
    thrust::stable_sort_by_key(d_dist_vec.begin(), d_dist_vec.end(), d_board_vec.begin());

    CUDA_SAFE_CALL(cudaFree(dev_distances));

    // removes the dupliates next to each other
    board_end = thrust::unique(d_board_vec.begin(), d_board_vec.end()); 
    d_board_vec.erase(board_end, d_board_vec.end());

    size_t size = d_board_vec.size();
    LOG_PRINTF("output size     : %d\n", size);
    
    Board *host_output = new Board[size];
    CUDA_SAFE_CALL(cudaMemcpy(host_output, &dev_boards[0], size * sizeof(Board),
                cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaFree(dev_boards));

    if (size > MAX_SIZE) {
        LOG_PRINTF("splitting...\n");
        for (int i = 0 ; i < size ; i += MAX_SIZE) {
            if (size < i + MAX_SIZE) {
                work_down(&host_output[i], size - i, !vertical, depth + 1);
            } else {
                work_down(&host_output[i], MAX_SIZE, !vertical, depth + 1);
            } 
        }
    } else {
        work_down(host_output, size, !vertical, depth + 1);
    }
    delete[] host_output;
}


// main routine that executes on the host
int main(void) {

    std::set_new_handler(outOfMemHandler);

    LOG_PRINTF("remember to kill the X session\n");
    LOG_PRINTF("Board size: %d\n", sizeof(Board));

    int inCount = 1;
    Board *inputBoards = new Board[inCount];
    memset(&inputBoards[0], 0, sizeof(Board));
    set_valid(&inputBoards[0], true);

    LOG_PRINTF("initial\n");
    print_board(&inputBoards[0], X_MAX, Y_MAX);
    work_down(inputBoards, 1, true, 0);
    delete[] inputBoards;
    return 0;
}
