#include <iostream>
#include <vector>
#include <map>
#include <queue>
#include <cstdlib>
#include <pthread.h>
#include <signal.h>
#include <time.h>
#include "board.h"


#ifdef DEBUG
#define DEBUG_MSG(str) do { std::cout << str << std::endl; } while( false )
#define DEBUG_BOARD(board) do { board->print(); } while ( false )
#else
#define DEBUG_MSG(str) do { } while ( false )
#define DEBUG_BOARD(board) do { } while ( false )
#endif

#define NUM_THREADS 5

using namespace std;

struct Input {
    Board *board;
    bool verticalMove;
};

//map<Board, Winner> known;

Winner solve(Board*, bool);

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

 bool operator ==(const Board &l, const Board &r) {
    if (!(l.x_max == r.y_max && l.y_max == r.y_max && l.player == r.player)) {
        return false;
    } else {
        int num = ceil((1.0 * l.x_max * l.y_max) / (8 * sizeof(uint64_t)));
        for (int i = 0 ; i < num ; i++) {
            if (l.bitboards[i] != r.bitboards[i]) {
                return false;
            }
        }
    }
    return true;
}

 bool operator < (const Board &l, const Board &r) {
    int num = ceil((1.0 * l.x_max * l.y_max) / (8 * sizeof(uint64_t)));
    for (int i = 0 ; i < num ; i++) {
        if (l.bitboards[i] != r.bitboards[i]) {
            return l.bitboards[i] < r.bitboards[i];
        }
    }
    return true;
}


void *solveThread(void *in) {
   struct Input *input = (struct Input*) in;
    Board *board = input->board;
    bool verticalMove = input->verticalMove;

    Winner winner = solve(board, verticalMove);

    delete board;
    delete input;

    if (winner == HORIZONTAL)
        pthread_exit((void *) 1);
    else
        pthread_exit((void *) 2);
}


Winner solve(Board *board, bool verticalMove) {
    Winner winner;

    DEBUG_BOARD(board);
    vector<int> *moves = board->next_moves(verticalMove);

    if (moves->size() == 0) {
        DEBUG_MSG("no more moves" << endl);
        if (verticalMove) {
            winner = HORIZONTAL;
        } else {
            winner = VERTICAL;
        }
    }

    for (unsigned int i = 0 ; i < moves->size() ; i++) {
        // place move
        //Board* newBoard = new Board(board);
        board->place_move(verticalMove, moves->at(i));

        winner = solve(board, !verticalMove);
        board->remove_move(verticalMove, moves->at(i));
        //delete newBoard;
        if ((verticalMove && winner == VERTICAL) ||
            (!verticalMove && winner == HORIZONTAL)) {
            break;
        }
    }
    delete moves;
    return winner;
}

Winner solve(Board *board) {
    Winner win1 = HORIZONTAL;
    Winner win2 = VERTICAL;

    vector<int> *moves = board->next_moves(true);
    int size = moves->size();
    pthread_t* threads = new pthread_t[size];

    for (int i = 0 ; i < size ; i++) {
        Board *newBoard = new Board(board);
        newBoard->place_move(true, moves->at(i));
        struct Input *input = new struct Input;
        input->board = newBoard;
        input->verticalMove = false;

        int rc = pthread_create(&threads[i], NULL, solveThread, (void *) input);
        if (rc) {
            cerr << "Error: unable to creat thread, " << rc << endl;
            exit(-1);
        }
    }
    for (int i = 0 ; i < size ; i++) {
        void* status;
        int rc = pthread_join(threads[i], &status);
        long s = (long) status;
        if (rc) {
            cerr << "Error: unable to join thread, " << rc << endl;
            exit(-1);
        }
        if (s == 2) {
            win1 = VERTICAL;
            cout << "exit early for vertical" << endl;
            break;
        }
    }
    delete moves;
    delete[] threads;

    moves = board->next_moves(false);
    size = moves->size();
    threads = new pthread_t[size];

    for (int i = 0 ; i < size ; i++) {
        Board *newBoard = new Board(board);
        newBoard->place_move(false, moves->at(i));
        struct Input *input = new struct Input;
        input->board = newBoard;
        input->verticalMove = true;

        int rc = pthread_create(&threads[i], NULL, solveThread, (void *) input);
        if (rc) {
            cerr << "Error: unable to creat thread, " << rc << endl;
            exit(-1);
        }
    }
    for (int i = 0 ; i < size ; i++) {
        void* status;
        int rc = pthread_join(threads[i], &status);
        long s = (long) status;
        if (rc) {
            cerr << "Error: unable to join thread, " << rc << endl;
            exit(-1);
        }
        if (s == 1) {
            win2 = HORIZONTAL;
            cout << "exit early for horizontal" << endl;
            break;
        }
    }
    delete moves;
    delete[] threads;

    print_winner(win1);
    print_winner(win2);

    if (win1 != win2) {
        if (win1 == VERTICAL) {
            return PLAYER_1;
        } else {
            return PLAYER_2;
        }
    } else {
        return win1;
    }
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

    Board* board = new Board(m, n);
    time_t start = time(NULL);
    Winner winner = solve(board);
    time_t after = time(NULL);
    print_winner(winner);
    cout << "seconds: " << (after - start) << endl;

    delete board;
    return 0;
}
