CUDA_INSTALL_PATH = /opt/cuda
TARGET = domineering
NVCC = nvcc
NVCCFLAGS = -I$(CUDA_INSTALL_PATH)/include -arch=compute_30 -Xcompiler -Wall

OBJECTS = $(patsubst %.cu, %.cu.o, $(wildcard *.cu))
HEADERS = $(wildcard *.h)

default: domineering
all: default

%.cu.o: %.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(TARGET): $(OBJECTS)
	$(NVCC) $(OBJECTS) -o $(TARGET)

clean:
	-rm -f *.o
	-rm -f $(TARGET)
