#pragma once
#include <vector>
#include "layers/nn_layer.hh"
#include "nn_utils/cost_f.hh"

class NeuralNetwork {
private:
	std::vector<NNLayer*> layers;
	Matrix error;

	Matrix Y;
	Matrix dY;
	float learning_rate;

public:
	NeuralNetwork(float learning_rate = 0.001);
	~NeuralNetwork();

	Matrix forward(Matrix X);
	void backprop(Matrix pred, Matrix target, CostFunction* function);

	void addLayer(NNLayer *layer);
	std::vector<NNLayer*> getLayers() const;

};