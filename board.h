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
    public:
        unsigned char x_max;
        unsigned char y_max;
        bool player;
        uint64_t *bitboards;
        Winner winner;

        int get_index(int, int);
        bool get_location(int x, int y);
        void set_location(int index);
        void clear_location(int index);

        Board() {
        }
        Board(Board*);
        Board(const unsigned char, const unsigned char);

        ~Board() {
            delete[] bitboards;
        }

        vector<int>* next_moves(bool);
        void place_move(bool, int);
        void remove_move(bool, int);

        void print();

        
};

