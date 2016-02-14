#include "board.h"

using namespace std;

inline uint64_t get_offset(int index, int boardIndex) {
    int bitNum = index - (boardIndex * 8 * sizeof(uint64_t));
    return pow (2, bitNum);
}

inline int Board::get_index(int x, int y) {
    return ((int) y_max) * x + y;
}

Board::Board(const unsigned char x, const unsigned char y) {
    valid = true;
    x_max = x;
    y_max = y;
    player = true;
    int num = ceil((1.0 * x * y) / (8 * sizeof(uint64_t)));
    bitboards = new uint64_t[num]; 
    for (int i = 0 ; i < num ; i++) {
        bitboards[i] = 0;
    }
}

Board::Board(Board* board) {
    valid = true;
    x_max = board->x_max;
    y_max = board->y_max;
    player = board->player;
    int num = ceil((1.0 * x_max * y_max) / (8 * sizeof(uint64_t)));
    bitboards = new uint64_t[num];
    memcpy(bitboards, board->bitboards, num * sizeof(uint64_t));
}

inline bool Board::get_location(int x, int y) {
    int index = get_index(x, y);
    int boardIndex = index / (8 * sizeof(uint64_t));
    uint64_t offset = get_offset(index, boardIndex);
    return (bitboards[boardIndex] & offset) != 0;
}

inline void Board::set_location(int index) {
    int boardIndex = index / (8 * sizeof(uint64_t));
    uint64_t offset = get_offset(index, boardIndex);
    bitboards[boardIndex] = bitboards[boardIndex] | offset;
}

inline void Board::clear_location(int index) {
    int boardIndex = index / (8 * sizeof(uint64_t));
    uint64_t offset = ~get_offset(index, boardIndex);
    bitboards[boardIndex] = bitboards[boardIndex]  & offset;
}

vector<int>* Board::next_moves(bool verticalMove) {
    vector<int> *moves = new vector<int>;
    if (verticalMove) {
        for (int x = 0 ; x < x_max -1 ; x++) {
            for (int y = 0 ; y < y_max ; y++) {
                if (!get_location(x, y) && !get_location(x + 1, y)) {
                    moves->push_back(get_index(x, y));
                }
            }
        }
    } else {
        for (int x = 0 ; x < x_max ; x++) {
            for (int y = 0 ; y < y_max - 1; y++) {
                if (!get_location(x, y) && !get_location(x, y + 1)) {
                    moves->push_back(get_index(x, y));
                }
            }
        }
    }
    cout << "size: " << moves->size() << endl;
    return moves;
}


void Board::place_move(bool verticalMove, int index) {
    set_location(index);
    if (verticalMove) {
        set_location(index + y_max);
    } else {
        set_location(index + 1);
    }
}

void Board::remove_move(bool verticalMove, int index) {
    clear_location(index);
    if (verticalMove) {
        clear_location(index + y_max);
    } else {
        clear_location(index + 1);
    }
}

void Board::print() {
    cout << "size: (" << (int) x_max << ", " << (int) y_max << ")" << endl;
    cout << "   ";
    for (int y = 0 ; y < y_max ; y++) {
        cout << y << "  ";
    }
    cout << endl;
    for (int x = 0 ; x < x_max ; x++) {
        cout << x << " ";
        for (int y = 0 ; y < y_max ; y++) {
            if (get_location(x, y) == true) {
                cout << " X ";
            } else {
                cout << " . ";
            }
        }
        cout << endl;
    }
}
