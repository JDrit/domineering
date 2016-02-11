#include <cstdlib>
#include <iostream>
#include <vector>
#include <math.h>
#include <stdint.h>
#include <string.h>

using namespace std;

typedef enum {
    NO_WINNER,
    PLAYER_1,
    PLAYER_2,
    VERTICAL,
    HORIZONTAL
} Winner;

class Board {
    unsigned char x_max;
    unsigned char y_max;
    bool player;
    uint64_t *bitboards;
    Winner winner;

    int get_index(int x, int y) {
        return ((int) y_max) * x + y;
    }

    bool get_location(int x, int y);

    public:
        Board (unsigned char, unsigned char);

        void set_location(int index);
        void clear_location(int index);

        vector<int>* next_moves(bool);
        place
}

int board_get_index(Board *board, int x, int y);

/**
 * If the location given by the board has a piece
 * set of not. Undefined if x >= x_max or y >= y_max
 */
bool board_get_location(Board *board, int x, int y);

/**
 * Generates all possible moves that the player can do.
 * The result if the location index representing the move.
 * If it is for the vertical player than it is the top point.
 * Else if it is the horizontal player, it is the left point.
 */
std::vector<int>* board_next_moves(Board *board, bool verticalMove);

/**
 * Places the move, depending on if it is a vetical or horizontal
 * move. The move int is either the top (vertical) spot or the
 * left (horizontal) spot.
 */
void board_place_move(Board *board, bool verticalMove, int move);

void board_remove_move(Board *board, bool verticalMove, int move);

Board* copy_board(Board *board);


void print_board(Board *board);

void board_free(Board *board);
