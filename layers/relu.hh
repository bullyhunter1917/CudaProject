#pragma once
#include "nn_layer.hh"

class ReLu : public NNLayer{
private:
    Matrix A;

    Matrix Z;
    Matrix dZ;
public:
    ReLu(std::string name);
    ~ReLu();

    Matrix& forward(Matrix& Z);
    Matrix& backprop(Matrix& dA, float learning_rate = 0.01);
};