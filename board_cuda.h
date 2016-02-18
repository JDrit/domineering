#ifndef board_cuda_h
#define board_cuda_h

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>

#define GET_INDEX(y_max, x, y) (y_max * x + y)

typedef struct {
    uint64_t bitboards[2];
    double distance;
} Board;

__host__ __device__ inline bool is_valid(const Board *board) {
    return board->bitboards[0] & 1 != 0;
}

__host__ __device__ void set_valid(Board *board, bool valid) {
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
        offset = (uint64_t) pow(2.0, index + 1);
    } else { // board 2
        boardIndex = 1;
        offset = (uint64_t) pow(2.0, index - 63);
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

#endif
