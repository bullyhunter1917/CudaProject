#include "softmax.hh"
#include <math.h>
#include <float.h>

__device__ float softmaxMaxValue(float* Z, int Z_x_dim, int Z_y_dim, int col) {
    float max_val = -FLT_MAX;

    for (int j = 0; j < Z_y_dim; j++) {
        if (Z[col + j*Z_x_dim] > max_val) {
            max_val = Z[col + j*Z_x_dim];
        }
    }

    return max_val;
}

__device__ float softmaxSumBatches(float* Z, int Z_x_dim, int Z_y_dim, int col) {
    float part_sum = 0.0f;
    float max_val = softmaxMaxValue(Z, Z_x_dim, Z_y_dim, col);

    for (int j = 0; j < Z_y_dim; j++) {
        part_sum += exp(Z[col + j*Z_x_dim] - max_val);
    }
    
    return part_sum;
}

__global__ void softmaxActivationForward(float* Z, float* A, int Z_x_dim, int Z_y_dim) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < Z_x_dim * Z_y_dim) {
        A[index] = exp(Z[index] - softmaxMaxValue(Z, Z_x_dim, Z_y_dim, index%Z_x_dim)) / softmaxSumBatches(Z, Z_x_dim, Z_y_dim, index%Z_x_dim);

        // printf("A: %f \n", A[index]);

    }

}

__global__ void softmaxActivationBackprop(float* A, float* dA, float* dZ, int A_x_dim, int A_y_dim) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < A_x_dim * A_y_dim) {
        dZ[index] = dA[index] * A[index] * (1 - A[index]);
        
        
        
        // float res = 0.0f;
        // for (int j = 0; j < A_y_dim; j++) {
        //     if (index == index%A_x_dim + j*A_x_dim) {
        //         res += dA[index%A_x_dim + j*A_x_dim] * A[index] * (1 - A[index]);
        //     } else {
        //         res += dA[index%A_x_dim + j*A_x_dim] * -1 * A[index] * A[index%A_x_dim + j*A_x_dim];
        //     } 
        // }
        // dZ[index] = res;
    }
}


SoftMax::SoftMax(std::string name) {
	this->name = name;
}

SoftMax::~SoftMax()
{ }

Matrix& SoftMax::forward(Matrix& Z) {
    this->Z = Z;
    A.allocateMemoryIfNotAllocated(Z.shape);
    dim3 block_size(256);
    dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x -1) / block_size.x);

    softmaxActivationForward<<<num_of_blocks, block_size>>>(Z.data_device.get(), A.data_device.get(), Z.shape.x, Z.shape.y);
    
    return A;
}

Matrix& SoftMax::backprop(Matrix& dA, float learning_rate) {
    dZ.allocateMemoryIfNotAllocated(Z.shape);

    dim3 block_size(256);
    dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x -1) / block_size.x);

    softmaxActivationBackprop<<<num_of_blocks, block_size>>>(A.data_device.get(), dA.data_device.get(), dZ.data_device.get(), A.shape.x, A.shape.y);
    return dZ;
}