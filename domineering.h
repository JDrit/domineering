#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <sys/time.h>
#include <iostream>

#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_free.h>
#include <thrust/copy.h>
#include <thrust/remove.h>

#include "board.h"

#define THREADS_PER_BLOCK 1024

#define CUDA_ERROR_CHECK
#define CUDA_SAFE_CALL( err ) __CUDA_SAFE_CALL( err, __FILE__, __LINE__ )
#define CUDA_CHECK_ERROR()    __CUDA_CHECK_ERROR( __FILE__, __LINE__ )
#define LOG_PRINTF(...) do { \
    struct timeval time_now; \
    gettimeofday(&time_now, NULL); \
    tm* time_str_tm = gmtime(&time_now.tv_sec); \
    printf("%02i:%02i:%02i:%06ld : ", time_str_tm->tm_hour, time_str_tm->tm_min, \
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


/**
 * Checks to see if the given call completed successfully.
 * Exits is there was an error.
 */
inline void __CUDA_SAFE_CALL(cudaError err, const char *file, const int line);

/**
 * Checks to see if there was a recent cuda error.
 * Exits is there was.
 */
inline void __CUDA_CHECK_ERROR(const char *file, const int line);

struct is_valid_struct {
    __host__ __device__ bool operator()(const Board b) {
        return !is_valid(&b);
    }
};

/**
 * Generates the next moves for the given input. This is the function that
 * is run on the GPU. Input is the pointer to the list of Boards and output
 * is where the next boards should be written to.
 */
__global__ void next_boards(Board *input, Board *output, int branching, int x_max, 
        int y_max, bool vertical);

void work_down(Board* dev_input, int x_max, int y_max, int inCount, bool vertical, int depth);
