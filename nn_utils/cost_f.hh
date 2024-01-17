#pragma once
#include "matrix.hh"

class CostFunction {

public:
    virtual ~CostFunction() = 0;

    virtual Matrix forward(Matrix pred, Matrix target, Matrix dY) = 0;
    virtual float cost(Matrix pred, Matrix target) = 0;
};

inline CostFunction::~CostFunction() {}