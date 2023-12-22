#include "linear_layer.hh"

__global__ void linearLayerForward(float* W, float* A, float* Z, float* b, int W_x_dim, int W_y_dim, int A_x_dim, int A_y_dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int Z_x_dim = A_x_dim;
    int Z_y_dim = W_y_dim;

    float Z_value = 0;

    if (row < Z_y_dim && col < Z_x_dim) {
        for (int i=0; i < W_x_dim; i++) {
            Z_value += W[row * W_x_dim + i] * A[i * A_x_dim + col];
        }
        Z[row * Z_x_dim + col] = Z_value + b[row];
    }
}

__global__ void linearLayerBackprop(float* W, float* dZ, float* dA, int W_x_dim, int W_y_dim, int dZ_x_dim, int dZ_y_dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int dA_x_dim = dZ_x_dim;
    int dA_y_dim = W_x_dim;

    float dA_value = 0.0f;

    if (row < dA_y_dim && col < dA_x_dim) {
        for (int i=0; i < W_y_dim; i++) {
            dA_value += W[i * W_x_dim + row] * dZ[i * dZ_x_dim + col];
        }
        dA[row * dA_x_dim + col] = dA_value;
    }
}

__global__ void linearLayerUpdateWeights(float* dZ, float* A, float* W, int dZ_x_dim, int dZ_y_dim, int A_x_dim, int A_y_dim, float learning_rate) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int W_x_dim = A_y_dim;
    int W_y_dim = dZ_y_dim;

    float dW_value = 0.0f;

    if (row < W_y_dim && col < W_x_dim) {
        for (int i=0; i < dZ_x_dim; i++) {
            dW_value += dZ[row * dZ_x_dim + i] * A[col * A_x_dim + i];
        }
        W[row * W_x_dim + col] = W[row * W_x_dim + col] - learning_rate * (dW_value / A_x_dim);
    }
}

__global__ void linearLayerUpdateBias(float* dZ, float* b, int dZ_x_dim, int dZ_y_dim, int b_x_dim, float learning_rate) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < dZ_x_dim * dZ_y_dim) {
        int dZ_x = index % dZ_x_dim;
        int dZ_y = index / dZ_x_dim;
        atomicAdd(&b[dZ_y, -learning_rate * (dZ[dZ_y * dZ_x_dim + dZ_x] / dZ_x_dim)]);
    }
}

void LinearLayer::initializeBiasWithZeros() {
    for (int x=0; x < b.shape.x; x++) {
        b[x] = 0;
    }

    b.copyHostToDevice();
}

void LinearLayer::initializeWeightsRandomly() {
    std::default_random_engine generator;
    std::normal_distribution<float> normal_distribution(0.0, 1.0);

    for (int x=0; x < W_shape.x; x++) {
        for (int y=0; y < W_shape.y; y++) {
            W[y * W_shape.x + x] = normal_distribution(generator) * weights_init_threshold;
        }
    }

    W.copyHostToDevice();
}

void LinearLayer::computeAndStoreBackpropError(Matrix& dZ) {
    dim3 block_size()

}

void LinearLayer::computeAndStoreLayerOutput(Matrix& A) {
    dim3 block_size(8, 8);
    dim3 num_of_blocks((Z.shape.x + block_size.x - 1) / block_size.x, (Z.shape.y + block_size.y - 1) / block_size.y);

    linearLayerForward<<<num_of_blocks, block_size>>>(W.data_device.get(), A.data_device.get(), Z.data_device.get(), b.data_device.get(), W.shape.x, W.shape.y, A.shape.x, A.shape.y);
}

void updateWeights(Matrix& dZ, float learning_rate) {
    dim3 block_size(8, 8);
    dim3 num_of_blocks((Z.shape.x + block_size.x - 1) / block_size.x, (Z.shape.y + block_size.y - 1) / block_size.y);
    
    linearLayerUpdateWeights<<<num_of_blocks, block_size>>>(dZ.data_device.get(), A.data_device.get(), W.data_device.get(), int dZ.shape.x, int dZ.shape.y, int A.shape.x, int A.shape.y, float learning_rate);
}

void updateBias(Matrix& dZ, float learning_rate) {
    linearLayerUpdateBias
}

LinearLayer::LinearLayer(std::string name, Shape W_shape) {
    this->name = name;
    this->W = Matrix(W_shape);
    this->b = Matrix(Shape(W_shape.y, 1));
    W.allocateMemory();
    b.allocateMemory();
    initializeWeightsRandomly();
    initializeBiasWithZeros();
}

Matrix& LinearLayer::forward(Matrix& A) {
    assert(W.shape.x == A.shape.y);

    this->A = A;
    Shape Z_shape(A.shape.x, W.shape.y);
    Z.allocateMemoryIfNotAllocated(Z_shape);

    computeAndStoreLayerOutput(A);

    return Z;
}

Matrix& LinearLayer::backprop(Matrix& dZ, float learning_rate) {
    dA.allocateMemoryIfNotAllocated(A.shape);

    computeAndStoreBackpropError(dZ);

    updateBias(dZ, learning_rate);

    updateWeights(dZ, learning_rate);

    return dA;
}

