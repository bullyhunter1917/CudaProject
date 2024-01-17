#pragma once

#include "dataset.hh"

class XOR : public Dataset{
private:
    size_t batch_size;
    size_t number_of_batches;

    std::vector<Matrix> batches;
    std::vector<Matrix> targets;
public:
    XOR(size_t batch_size, size_t number_of_batches);
    ~XOR();

    int getNumOfBatches();
    std::vector<Matrix>& getBatches();
    std::vector<Matrix>& getTargets();
};

