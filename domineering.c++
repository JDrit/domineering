#include <iostream>
#include <vector>
#include <map>
#include <cstdlib>
#include "board.h"

#ifdef DEBUG
#define DEBUG_MSG(str) do { std::cout << str << std::endl; } while( false )
#define DEBUG_BOARD(board) do { print_board(board) } while ( false )
#else
#define DEBUG_MSG(str) do { } while ( false )
#define DEBUG_BOARD(board) do { } while ( false )
#endif

using namespace std;

map<Board, Winner> known;

void print_winner(Winner winner) {
    switch (winner) {
        case VERTICAL:
            cout << "winner: vertical player" << endl;
            break;
        case HORIZONTAL:
            cout << "winner: horizontal player" << endl;
            break;
        case PLAYER_1:
            cout << "winner: player 1" << endl;
            break;
        case PLAYER_2:
            cout << "winner: player 2" << endl;
            break;
        case NO_WINNER:
            cout << "no winner" << endl;
            break;
        default:
            cout << "winner undefined" << endl;
    }
}

Winner solve(Board *board, bool verticalMove) {
    Winner winner;

    DEBUG_BOARD(board);
    vector<int> *moves = board_next_moves(board, verticalMove);

    if (moves->size() == 0) {
        DEBUG_MSG("no more moves" << endl);
        if (verticalMove) {
            winner = HORIZONTAL;
        } else {
            winner = VERTICAL;
        }
    }

    for (int i = 0 ; i < moves->size() ; i++) {
        // place move
        Board* newBoard = copy_board(board);
        board_place_move(newBoard, verticalMove, moves->at(i));

        Board b = *board;
        map<Board, Winner>::iterator it = known.find(b);
        if (it != known.end()) {
            cout << "found" << endl;
            Winner found = it->second;
            print_winner(found);
            return found;
        } else {
            winner = solve(newBoard, !verticalMove);
            if ((verticalMove && winner == VERTICAL) ||
                (!verticalMove && winner == HORIZONTAL)) {
                break;
            }

        }
    }
    delete moves;
    pair<Board, Winner> pair = make_pair(board, winner);
    known.insert(pair);
    return winner;
}

int main(int argc, char* argv[]) {
    int m, n;

    if (argc < 3) {
        cerr << "enter in board dimensions: " << argv[0] << " m n" << endl;
        exit(1);
    }
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    DEBUG_MSG("generating board for " << m << "x" << n << endl);

    Board* board = generate_board(m, n);

    Winner winner = solve(board, true);
    print_winner(winner);

    board_free(board);
    return 0;
}
