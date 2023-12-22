#include <iostream>
#include "relu.hh"

__device__ float relu(float x) {
    if (x > 0) {
        return x;
    }
    else {
        return 0.0;
    }
}

__device__ float reluback(float x) {
    if (x > 0) {
        return 1.0;
    }
    else {
        return 0.0;
    }
}

__global__ void reluActivationForward(float* Z, float* A, int Z_x_dim, int Z_y_dim) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < Z_x_dim * Z_y_dim) {
        A[index] = relu(Z[index]);
    }
}

__global__ void reluActivationBackprop(float* Z, float* dA, float* dZ, int Z_x_dim, int Z_y_dim) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < Z_x_dim * Z_y_dim) {
        dZ[index] = dA[index] * reluback(Z[index]);
    }
}

ReLu::ReLu(std::string name) {
    this->name = name;
    this->A = Matrix();
    this->Z = Matrix();
    this->dZ = Matrix();
}

Matrix& ReLu::forward(Matrix& Z) {
    this->Z = Z;
    A.allocateMemoryIfNotAllocated(Z.shape);

    dim3 block_size(256);
    dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);

    reluActivationForward<<<num_of_blocks, block_size>>>(Z.data_device.get(), A.data_device.get(), Z.shape.x, Z.shape.y);

    return A;
}

Matrix& ReLu::backprop(Matrix& dA, float learning_rate) {
    dZ.allocateMemoryIfNotAllocated(dA.shape);

    dim3 block_size(256);
    dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);

    reluActivationBackprop<<<num_of_blocks, block_size>>>(Z.data_device.get(), dA.data_device.get(), dZ.data_device.get(), Z.shape.x, Z.shape.y);

    return dZ;
}