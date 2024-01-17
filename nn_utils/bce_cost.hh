#pragma once
#include "cost_f.hh"

class BCECost : public CostFunction {
public:
    ~BCECost();
    Matrix forward(Matrix pred, Matrix target, Matrix dY);
    float cost(Matrix pred, Matrix target);
};