INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart

CC := nvcc

NVCCFLAGS	:= -lineinfo -arch=sm_61 --ptxas-options=-v --use_fast_math

FOLDERS := 

all:
	$(MAKE) -C nn_utils
	$(MAKE) -C layers
	nvcc main.cu -o main layers/sigmoid.o layers/relu.o nn_utils/shape.o nn_utils/matrix.o
	
		
clean:
	$(MAKE) -C nn_utils clean
	$(MAKE) -C layers clean
	
