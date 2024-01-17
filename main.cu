#include <iostream>
#include "neural_network.hh"
#include "layers/linear_layer.hh"
#include "layers/sigmoid.hh"
#include "layers/relu.hh"
#include "layers/softmax.hh"
#include "nn_utils/bce_cost.hh"
#include "nn_utils/ce_cost.hh"
#include "xor.hh"
#include "mnist.hh"



int main(int argc, char const *argv[])
{
    MNIST ds(2);

    // Printing how data looks

    // Matrix dane;
    // Matrix tar;
    // Matrix lab;

    // dane = ds.getBatches().at(1);
    // tar = ds.getTargets().at(1);
    // lab = ds.getLabels().at(1);

    // std::cout << dane[0] << ", " << dane[784] << std::endl;

    // std::cout << lab[0] << ", " << lab[1] << std::endl;

    // std::cout << tar.shape.x << ", " << tar.shape.y << std::endl;
    // for (int i=0; i < 20; i=i+2) {
    //     std::cout << "Dane: " << tar[i] << "," << tar[i+1] << std::endl;
    // }

    CECost ce_cost;

    NeuralNetwork nn;
    nn.addLayer(new LinearLayer("l_1", Shape(784, 128)));
    nn.addLayer(new ReLu("r1"));
    nn.addLayer(new LinearLayer("l_2", Shape(128, 64)));
    nn.addLayer(new ReLu("r2"));
    nn.addLayer(new LinearLayer("l_3", Shape(64, 10)));
    nn.addLayer(new SoftMax("softmax"));

    Matrix Y;
    for (int epoch = 0; epoch < 46; epoch++) {
        float cost = 0.0f;

        for (int batch = 0; batch < ds.getNumOfBatches() - 1; batch++) {
            Y = ds.getBatches().at(batch);
            // std::cout << Y[0] << " " << Y[784] << std::endl;
            Y = nn.forward(ds.getBatches().at(batch));
            // std::cout << Y[0] << " " << Y[784] << std::endl;
            nn.backprop(Y, ds.getTargets().at(batch), &ce_cost);
            cost += ce_cost.cost(Y, ds.getTargets().at(batch));
        }

        if (epoch % 1 == 0) {
            std::cout << "Epoch: " << epoch << ", Cost: " << cost / ds.getNumOfBatches() << std::endl;
        }
    }
    
    Y = nn.forward(ds.getBatches().at(0));
    Y.copyDeviceToHost();
    
    Matrix lab;
    lab = ds.getLabels().at(0);
    std::cout << "Outputu size: " << Y.shape.x << ", " << Y.shape.y << std::endl;
    std::cout << "Expected value" << lab[0] << ", " << lab[1] << std::endl;
    for (int i=0; i < 20; i=i+2) {
        std::cout << "Numer: " << i/2 << " Dane: " << double(Y[i]) << "," << double(Y[i+1]) << std::endl;
    }


    // Matrix input_data = Matrix(Shape(4,2));
    // input_data.allocateMemory();
    // input_data[0] = 0.0f;
    // input_data[4] = 0.0f;
    // input_data[1] = 0.0f;
    // input_data[5] = 1.0f;
    // input_data[2] = 1.0f;
    // input_data[6] = 0.0f;
    // input_data[3] = 1.0f;
    // input_data[7] = 1.0f;
    // input_data.copyHostToDevice();

    // XOR dataset(3, 4);

    // BCECost bce_cost;

    // NeuralNetwork nn;
    // nn.addLayer(new LinearLayer("l_1", Shape(2, 2)));
    // nn.addLayer(new Sigmoid("s_1"));
    // nn.addLayer(new LinearLayer("l_2", Shape(2, 1)));
    // nn.addLayer(new Sigmoid("s_output"));

    

    // Matrix Y;
    // for (int epoch = 0; epoch < 10001; epoch++) {
    //     float cost = 0.0;

    //     for (int batch = 0; batch < dataset.getNumOfBatches() - 1; batch++) {
    //         //std::cout << dataset.getBatches().at(batch)[0] << ", " << dataset.getBatches().at(batch)[3] << ' ' << dataset.getTargets().at(batch)[0] << std::endl;
    //         Y = nn.forward(dataset.getBatches().at(batch));
    //         nn.backprop(Y, dataset.getTargets().at(batch), &bce_cost);
    //         cost += bce_cost.cost(Y, dataset.getTargets().at(batch));
    //     }
        
    //     if (epoch % 1000 == 0) {
    //         std::cout << "Epoch: " << epoch << ", Cost: " << cost / dataset.getNumOfBatches() << std::endl;
    //     }
    // }

    // Y = nn.forward(input_data);
    // Y.copyDeviceToHost();
    // std::cout << Y.shape.x << ", " << Y.shape.y << std::endl;
    // for (int i=0; i < 4; i++) {
    //     std::cout << "Dane: " << input_data[i] << "," << input_data[i+input_data.shape.x] << " Result: " << Y[i] << std::endl;
    // }
}
