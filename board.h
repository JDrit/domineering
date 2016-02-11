#include <cstdlib>
#include <stdint.h>
#include <math.h>

typedef struct {
    unsigned char x_max;
    unsigned char y_max;
    bool player;
    uint64_t *bitboards;
    bool winner;
} Board;


bool board_get_location(Board *board, int x, int y);

void board_set_location(Board *board, int x, int y);

Board* generate_board(int x, int y);

void print_board(Board *board);

void board_free(Board *board);
