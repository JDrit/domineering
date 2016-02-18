#ifndef board_cuda_h
#define board_cuda_h

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>

typedef struct {
    uint64_t bitboards[2];
} Board;

__host__ __device__ inline bool is_valid(const Board *board);

__host__ __device__ inline void set_valid(Board *board, bool valid);

inline bool boards_equal(const Board *b1, const Board *b2);

__device__ bool operator ==(const Board& b1, const Board& b2);

int compare_boards(const void *v1, const void *v2);

__device__ bool operator <(const Board& b1, const Board &b2);

__host__ __device__ inline bool get_location(Board *board, int y_max, int x, int y);

__host__ __device__ inline void set_location(Board *board, int y_max, int x, int y);

__host__ __device__ void print_board(Board *board, int x_max, int y_max);

/**
 * Flips the board so that there are more pieces on the left side than
 * the right. This is done so the boards that are symetrical across
 * the vertical axis get deduplicated.
 */
__device__ inline void copy_left(Board *board, int x_max, int y_max);



#endif
