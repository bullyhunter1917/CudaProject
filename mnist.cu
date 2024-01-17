#include "mnist.hh"
#include <fstream>
#include <iostream>
#include <string>

uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

MNIST::MNIST(size_t batch_size, size_t number_of_batches) :
    batch_size(batch_size), number_of_batches(number_of_batches)
{
    std::ifstream image_file("datasets/mnist/train-images.idx3-ubyte", std::ios::in | std::ios::binary);
    std::ifstream label_file("datasets/mnist/train-labels.idx1-ubyte", std::ios::in | std::ios::binary);

    uint32_t magic = 0;
    uint32_t num_items = 0;
    uint32_t num_labels = 0;

    image_file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    if(magic != 2051) {
        printf("Incorect image file magic \n");
        return;
    }

    label_file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    if(magic != 2049){
        printf("Incorect image file magic \n");
        return;
    }

    image_file.read(reinterpret_cast<char*>(&num_items), 4);
    num_items = swap_endian(num_items);
    label_file.read(reinterpret_cast<char*>(&num_labels), 4);
    num_labels = swap_endian(num_labels);
    printf("num_items: %i, num_labels: %i \n", num_items, num_labels);
    if(num_items != num_labels) {
        printf("Image file nums should equal to label num \n");
        return;
    }

    for (int i = 0; i < number_of_batches; i++) {
        batches.push_back(Matrix(Shape(batch_size, 784)));
        targets.push_back(Matrix(Shape(batch_size, 10)));
        labels.push_back(Matrix(Shape(batch_size, 1)));

        batches[i].allocateMemory();
        targets[i].allocateMemory();
        labels[i].allocateMemory();

        for (int k = 0; k < batch_size; k++) {
            char a;
            label_file.read(&a, 1);
            
            labels[i][k] = int(a);

            for (int number = 0; number < 10; number++) {
                if(number == int(a)) {
                    targets[i][k + (targets[i].shape.x * number)] = 1;
                } else {
                    targets[i][k + (targets[i].shape.x * number)] = 0;
                }
            }
            
            for (int pixel_ind = 0; pixel_ind < 784; pixel_ind++) {
                char b;
                image_file.read(&b, 1);
                batches[i][k + (batches[i].shape.x * pixel_ind)] = float(uint8_t(b))/255.0;
            }
        }

        batches[i].copyHostToDevice();
        targets[i].copyHostToDevice();
        labels[i].copyHostToDevice();
    }
}

MNIST::MNIST(size_t batch_size) :
    MNIST(batch_size, 10000/batch_size)
{ }

int MNIST::getNumOfBatches() {
    return number_of_batches;
}

std::vector<Matrix>& MNIST::getBatches() {
    return batches;
}

std::vector<Matrix>& MNIST::getTargets() {
    return targets;
}

std::vector<Matrix>& MNIST::getLabels() {
    return labels;
}

MNIST::~MNIST() {}