#include "ce_cost.hh"

#include <assert.h>

__global__ void crossEntropyCost(float* pred, float* target, int size, float* cost) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        float partial_cost = target[index] * logf(pred[index]);
        atomicAdd(cost, - partial_cost / size);
    }
}

__global__ void dCrossEntropyCost(float* pred, float* target, float* dY, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        dY[index] = -1 * (target[index]/pred[index]) + (1 - target[index])/(1 - pred[index]);
    }
}

float CECost::cost(Matrix pred, Matrix target) {
    assert(pred.shape.x == target.shape.x);

    float* cost;
    cudaMallocManaged(&cost, sizeof(float));
    *cost = 0.0f;

    dim3 block_size(256);
    dim3 num_of_blocks((pred.shape.x + block_size.x - 1) / block_size.x);
    crossEntropyCost<<<num_of_blocks, block_size>>>(pred.data_device.get(),
														  target.data_device.get(),
														  pred.shape.x * pred.shape.y, cost);
	cudaDeviceSynchronize();

    float cost_value = *cost;
    cudaFree(cost);

    return cost_value;
}

Matrix CECost::forward(Matrix pred, Matrix target, Matrix dY) {
    assert(pred.shape.x == target.shape.x);

    dim3 block_size(256);
    dim3 num_of_blocks((pred.shape.x + block_size.x - 1) / block_size.x);
    dCrossEntropyCost<<<num_of_blocks, block_size>>>(pred.data_device.get(), target.data_device.get(), dY.data_device.get(), pred.shape.x*pred.shape.y);

    return dY;
}

CECost::~CECost()
{ }