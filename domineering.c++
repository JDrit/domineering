#define BOOST_THREAD_PROVIDES_FUTURE
#define BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION

#include <iostream>
#include <vector>
#include <map>
#include <queue>
#include <cstdlib>
#include <pthread.h>
#include <signal.h>
#include <time.h>
#include "board.h"
#include <boost/thread.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/future.hpp>



#ifdef DEBUG
#define DEBUG_MSG(str) do { std::cout << str << std::endl; } while( false )
#define DEBUG_BOARD(board) do { board->print(); } while ( false )
#else
#define DEBUG_MSG(str) do { } while ( false )
#define DEBUG_BOARD(board) do { } while ( false )
#endif

#define HORIZONTAL_WIN 1
#define VERTICAL_WIN 2
#define PLAYER_1 3
#define PLAYER_2 4

using namespace std;
using namespace boost;

struct Input {
    Board *board;
    bool verticalMove;
};

//map<Board, Winner> known;

Winner solve(Board*, bool);

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


void solveThread(void* in, boost::promise<int> &p) {
    struct Input *input = (struct Input*) in;
    Board *board = input->board;
    bool verticalMove = input->verticalMove;

    Winner winner = solve(board, verticalMove);

    delete board;
    delete input;

    if (winner == HORIZONTAL)
        p.set_value(HORIZONTAL_WIN);
    else
        p.set_value(VERTICAL_WIN);
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

int solveThreadPool(Board *board, bool vertical) {
    int winner;

    if (vertical)
        winner = HORIZONTAL_WIN;
    else
        winner = VERTICAL_WIN;

    vector<int> *moves = board->next_moves(vertical);
    int size = moves->size();
    boost::thread_group thread_group;
    boost::promise<int> *promises = new boost::promise<int>[size];

    for (int i = 0 ; i < size ; i++) {
        Board *newBoard = new Board(board);
        newBoard->place_move(vertical, moves->at(i));
        struct Input *input = new struct Input;
        input->board = newBoard;
        input->verticalMove = !vertical;

        thread_group.add_thread(new boost::thread(solveThread, input, std::ref(promises[i])));
    }

    for (int i = 0 ; i < size ; i++) {
        boost::future<int> f = promises[i].get_future();
        int win = f.get();
        if (win == VERTICAL_WIN) {
            winner = VERTICAL_WIN;
            DEBUG_MSG("exiting early for vertical");
            break;
        }
    }
    thread_group.join_all();
    delete[] promises;
    delete moves;
    return winner;
}

int solve(Board *board) {
    int win1 = solveThreadPool(board, true);
    int win2 = solveThreadPool(board, false);

    if (win1 != win2) {
        if (win1 == VERTICAL_WIN) {
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
    int winner = solve(board);
    time_t after = time(NULL);
    cout << "winner: " << winner << endl;
    cout << "took seconds: " << (after - start) << endl;
    delete board;
    return 0;
}
