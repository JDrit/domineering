#include "domineering.hpp"

using namespace std;
using namespace boost;

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

    int winner = solve(board, verticalMove);

    delete board;
    delete input;

    p.set_value(winner);
}


int solve(Board *board, bool verticalMove) {
    int winner;

    DEBUG_BOARD(board);
    vector<int> *moves = board->next_moves(verticalMove);

    if (moves->size() == 0) {
        DEBUG_MSG("no more moves" << endl);
        if (verticalMove) {
            winner = HORIZONTAL_WIN;
        } else {
            winner = VERTICAL_WIN;
        }
    }

    for (unsigned int i = 0 ; i < moves->size() ; i++) {
        // place move
        board->place_move(verticalMove, moves->at(i));

        winner = solve(board, !verticalMove);
        board->remove_move(verticalMove, moves->at(i));
        //delete newBoard;
        if ((verticalMove && winner == VERTICAL_WIN) ||
            (!verticalMove && winner == HORIZONTAL_WIN)) {
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
