#pragma once
#include "nn_utils/matrix.hh"
#include <vector>

class Dataset {
public:
    virtual ~Dataset() = 0;

    virtual int getNumOfBatches() = 0;
    virtual std::vector<Matrix>& getBatches() = 0;
    virtual std::vector<Matrix>& getTargets() = 0;
};

inline Dataset::~Dataset()
{ }