#include "board.h"

using namespace std;

inline uint64_t get_offset(int index) {
    int boardIndex = index / (8 * sizeof(uint64_t));
    int bitNum = index - (boardIndex * 8 * sizeof(uint64_t));
    return pow (2, bitNum);
}

Board::Board(unsigned char x, unsigned char y) {
    x_max = x;
    y_max = y;
    player = true;
    int num = ceil((1.0 * x * y) / (8 * sizeof(uint64_t)));
    bitboards = (uint64_t*) malloc(num * sizeof(uint64_t));
    for (int i = 0 ; i < num ; i++) {
        bitboards[i] = 0;
    }
    winner = NO_WINNER;
}

void Board::set_location(int index) {
    int boardIndex = index / (8 * sizeof(uint64_t));
    uint64_t offset = get_offset(index);
    bitboards[boardIndex] = bitboards[boardIndex] | offset;
}

void Board::clear_location(int index) {
    int boardIndex = index / (8 * sizeof(uint64_t));
    uint64_t offset = ~get_offset(index);
    bitboards[boardIndex] = bitboards[boardIndex]  & offset;
}

vector<int>* Board::next_moves(bool verticalMove) {
   vector<int> *moves = new vector<int>;

    if (verticalMove) {
        for (int x = 0 ; x < board->x_max -1 ; x++) {
            for (int y = 0 ; y < board->y_max ; y++) {
                if (!board_get_location(board, x, y) && !board_get_location(board, x + 1, y)) {
                    moves->push_back(board_get_index(board, x, y));
                }
            }
        }
    } else {
        for (int x = 0 ; x < board->x_max ; x++) {
            for (int y = 0 ; y < board->y_max - 1; y++) {
                if (!board_get_location(board, x, y) && !board_get_location(board, x, y + 1)) {
                    moves->push_back(board_get_index(board, x, y));
                }
            }
        }
    }

    return moves;
}


void board_place_move(Board *board, bool verticalMove, int index) {
    board_set_location(board, index);
    if (verticalMove) {
        board_set_location(board, index + board->y_max);
    } else {
        board_set_location(board, index + 1);
    }
}

void board_remove_move(Board *board, bool verticalMove, int index) {
    board_clear_location(board, index);
    if (verticalMove) {
        board_clear_location(board, index + board->y_max);
    } else {
        board_clear_location(board, index + 1);
    }
}

bool board_get_location(Board *board, int x, int y) {
    int index = board_get_index(board, x, y);
    int boardIndex = index / (8 * sizeof(uint64_t));
    uint64_t offset = get_offset(index);
    return (board->bitboards[boardIndex] & offset) != 0;
}

Board* copy_board(Board *board) {
    Board* newBoard = (Board*) malloc(sizeof(Board));
    newBoard->x_max = board->x_max;
    newBoard->y_max = board->y_max;
    newBoard->player = board->player;
    int num = ceil((1.0 * board->x_max * board->y_max) / (8 * sizeof(uint64_t)));
    newBoard->bitboards = (uint64_t*) malloc(num * sizeof(uint64_t));
    memcpy(newBoard->bitboards, board->bitboards, num * sizeof(uint64_t));
    return newBoard;
}

void print_board(Board *board) {
    cout << "size: (" << (int) board->x_max << ", " << (int) board->y_max << ")" << endl;
    cout << "   ";
    for (int y = 0 ; y < board->y_max ; y++) {
        cout << y << "  ";
    }
    cout << endl;
    for (int x = 0 ; x < board->x_max ; x++) {
        cout << x << " ";
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


