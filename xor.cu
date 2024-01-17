#include "xor.hh"
#include <cstdio>

XOR::XOR(size_t batch_size, size_t number_of_batches):
    batch_size(batch_size), number_of_batches(number_of_batches)
{
    int numbers[8] = {0, 1, 0, 1, 0, 0, 1, 1};
    int result[4] = {0, 1, 1, 0};
    int pntr = 0;
    for (int i = 0; i < number_of_batches; i++) {
        batches.push_back(Matrix(Shape(batch_size, 2)));
        targets.push_back(Matrix(Shape(batch_size, 1)));

        batches[i].allocateMemory();
        targets[i].allocateMemory();

        for (int k = 0; k < batch_size; k++) {
            printf("ptr: %i, first: %i, sec: %i, tar: %i \n", pntr, numbers[pntr], numbers[pntr+4], result[pntr]);
            batches[i][k] = numbers[pntr];
            batches[i][batch_size + k] = numbers[pntr + 4];

            targets[i][k] = result[pntr];

            pntr = (pntr + 1) % 4;
        }

        batches[i].copyHostToDevice();
        targets[i].copyHostToDevice();
    }
}

int XOR::getNumOfBatches() {
    return number_of_batches;
}

std::vector<Matrix>& XOR::getBatches() {
    return batches;
}

std::vector<Matrix>& XOR::getTargets() {
    return targets;
}

XOR::~XOR() {}