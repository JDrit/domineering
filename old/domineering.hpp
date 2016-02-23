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

struct Input {
    Board *board;
    bool verticalMove;
};

void solveThread(void* in, boost::promise<int> &p);


int solve(Board *board, bool verticalMove);


int solveThreadPool(Board *board, bool vertical);


int solve(Board *board);
