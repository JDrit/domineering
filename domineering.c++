#include <iostream>
#include <cstdlib>
#include "board.h"
using namespace std;

int main(int argc, char* argv[]) {
    int m, n;

    if (argc < 3) {
        cerr << "enter in board dimensions: " << argv[0] << " m n" << endl;
        exit(1);
    }
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    cout << "generating board for " << m << "x" << n << endl;

    Board* board = generate_board(m, n);

    print_board(board);
    board_set_location(board, 5, 3);
    print_board(board);

    board_free(board);
    return 0;
}
