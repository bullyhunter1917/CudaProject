#include "nn_layer.hh"

class Sigmoid : public NNLayer {
private:
    Matrix A;

    Matrix Z;
    Matrix dZ;
public:
    Sigmoid(std::string name);
    ~Sigmoid();

    Matrix& forward(Matrix& Z);
    Matrix& backprop(Matrix& dA, float learning_rate = 0.01);
};