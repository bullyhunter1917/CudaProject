INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart

CC := nvcc

NVCCFLAGS	:= -lineinfo -arch=sm_61 --ptxas-options=-v --use_fast_math

FOLDERS := 

all:
	$(MAKE) -C nn_utils
	$(MAKE) -C layers
	nvcc -c xor.cu -o xor.o
	nvcc -c mnist.cu -o mnist.o
	nvcc -c neural_network.cu -o neural_network.o
	nvcc main.cu -o main layers/sigmoid.o layers/relu.o layers/linear_layer.o layers/softmax.o nn_utils/shape.o nn_utils/matrix.o nn_utils/bce_cost.o nn_utils/ce_cost.o neural_network.o xor.o mnist.o
	
		
clean:
	$(MAKE) -C nn_utils clean
	$(MAKE) -C layers clean
	
