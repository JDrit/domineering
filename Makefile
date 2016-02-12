TARGET = domineering
LIBS = -lm -pthread -lboost_system -lboost_thread
CC = clang++
CFLAGS = -g -Wall -std=c++11

.PHONY: default all clean

default: $(TARGET)
all: default

OBJECTS = $(patsubst %.c++, %.o, $(wildcard *.c++))
HEADERS = $(wildcard *.h)

%.o: %.c++ $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

.PRECIOUS: $(TARGET) $(OBJECTS)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -Wall $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f $(TARGET)
