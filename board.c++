#include <iostream>
#include "board.h"

using namespace std;

inline int board_get_index(Board *board, int x, int y) {
    return ((int) board->y_max) * x + y;
}

inline uint64_t get_offset(int index) {
    int boardIndex = index / (8 * sizeof(uint64_t));
    int bitNum = index - (boardIndex * 8 * sizeof(uint64_t));
    return pow (2, bitNum);
}

bool board_get_location(Board *board, int x, int y) {
    int index = board_get_index(board, x, y);
    int boardIndex = index / (8 * sizeof(uint64_t));
    uint64_t offset = get_offset(index);
    return (board->bitboards[boardIndex] & offset) != 0;
}


void board_set_location(Board *board, int x, int y) {
    int index = board_get_index(board, x, y);
    int boardIndex = index / (8 * sizeof(uint64_t));
    uint64_t offset = get_offset(index);
    board->bitboards[boardIndex] = board->bitboards[boardIndex] | offset;
}

Board* generate_board(int x, int y) {
    Board* board = (Board*) malloc(sizeof(Board));
    board->x_max = x;
    board->y_max = y;
    board->player = true;
    int num = ceil((1.0 * x * y) / (8 * sizeof(uint64_t)));
    board->bitboards = (uint64_t*) malloc(num * sizeof(uint64_t));
    for (int i = 0 ; i < num ; i++) {
        board->bitboards[i] = 0;
    }
    return board;
}

void print_board(Board *board) {
    cout << "size: (" << (int) board->x_max << ", " << (int) board->y_max << ")" << endl;
    for (int x = 0 ; x < board->x_max ; x++) {
        for (int y = 0 ; y < board->y_max ; y++) {
            if (board_get_location(board, x, y) == true) {
                cout << " X ";
            } else {
                cout << " . ";
            }
        }
        cout << endl;
    }
}

void board_free(Board *board) {
    free(board->bitboards);
    free(board);
}



