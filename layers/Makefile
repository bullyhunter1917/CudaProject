INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart

CC := nvcc
CFLAGS	:= -lineinfo -arch=sm_61 --ptxas-options=-v --use_fast_math

SOURCES := $(*.cu)

SRCS=$(wildcard *.cu)
all: $(SRCS:.cu=.o)
	@echo "DONE"

%.o: %.cu
	$(CC) -c $(CFLAGS) $<

clean:
	rm -rf ./*.o
	
