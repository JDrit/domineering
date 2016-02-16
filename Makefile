CUDA_INSTALL_PATH = /opt/cuda
TARGET = domineering
LIBS = -lm -pthread -lboost_system -lboost_thread
CC = clang++
CFLAGS = -g -Wall -std=c++11

NVCC = nvcc
NVCCFLAGS = -I$(CUDA_INSTALL_PATH)/include -arch=compute_30
CUDA_OUT = cuda

.PHONY: default all clean

default: cuda
all: default

OBJECTS = $(patsubst %.c++, %.o, $(wildcard *.c++))
HEADERS = $(wildcard *.h)

%.o: %.c++ $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

%.cu.o: %.cu
	$(NVCC) -G -g $(NVCCFLAGS) -c $< -o $@

cuda: cuda.cu.o board.o
	$(NVCC) -Xcompiler "-std=c++0x" -G -g cuda.cu.o board.o -o $(CUDA_OUT)

.PRECIOUS: $(TARGET) $(OBJECTS)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -Wall $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f $(TARGET)
	-rm -r $(CUDA_OUT)
