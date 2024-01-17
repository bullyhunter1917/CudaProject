#pragma once

#include "dataset.hh"

class MNIST : public Dataset {
private:
    size_t batch_size;
    size_t number_of_batches;

    std::vector<Matrix> batches;
    std::vector<Matrix> targets;
    std::vector<Matrix> labels;
    
public:
    MNIST(size_t batch_size, size_t number_of_batches);
    MNIST(size_t batch_size);
    ~MNIST();

    int getNumOfBatches();
    std::vector<Matrix>& getBatches();
    std::vector<Matrix>& getTargets();
    std::vector<Matrix>& getLabels();
};