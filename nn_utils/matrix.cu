#include "matrix.hh"

void Matrix::allocateHostMemory() {
    if (!host_allocated) {
        data_host = std::shared_ptr<float>(new float[shape.x, shape.y], [&](float* ptr){ delete[] ptr; });
        host_allocated = true;
    }
}

void Matrix::allocateCudaMemory() {
    if (!device_allocated)
    {
        float* device_memory = nullptr;
        cudaMalloc(&device_memory, shape.x * shape.y * sizeof(float));
        //NNException::throwIfDeviceErrorsOccurred("Cannot allocate CUDA memory for Tensor3D");
        data_device = std::shared_ptr<float>(device_memory, [&](float* ptr){ cudaFree(ptr); });
        device_allocated = true;
    }
}

Matrix::Matrix(size_t x_dim, size_t y_dim) {
    this->shape = Shape(x_dim, y_dim);
    this->data_device = nullptr;
    this->data_host = nullptr;
    this->device_allocated = false;
    this->host_allocated = false;
}

Matrix::Matrix(Shape shape) {
    Matrix(shape.x, shape.y);
}

void Matrix::allocateMemory() {
    allocateHostMemory();
    allocateCudaMemory();
}

void Matrix::allocateMemoryIfNotAllocated(Shape shape) {
    this->shape.x = shape.x;
    this->shape.y = shape.y;

    allocateMemory();
}