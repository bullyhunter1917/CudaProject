#pragma once
#include "cost_f.hh"

class CECost : public CostFunction {
public:
    ~CECost();
    Matrix forward(Matrix pred, Matrix target, Matrix dY);
    float cost(Matrix pred, Matrix target);
};