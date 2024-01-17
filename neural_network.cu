#include "neural_network.hh"

NeuralNetwork::NeuralNetwork(float learning_rate) {
    this->learning_rate = learning_rate;
}

NeuralNetwork::~NeuralNetwork() {
	for (auto layer : layers) {
		delete layer;
	}
}

void NeuralNetwork::addLayer(NNLayer* layer) {
	this->layers.push_back(layer);
}

Matrix NeuralNetwork::forward(Matrix X) {
	Matrix Z = X;

	for (auto layer : layers) {
		
		Z = layer->forward(Z);
	}

	Y = Z;
	return Y;
}

void NeuralNetwork::backprop(Matrix pred, Matrix target, CostFunction* function) {
	dY.allocateMemoryIfNotAllocated(pred.shape);
	Matrix err = function->forward(pred, target, dY);

	for (auto it = this->layers.rbegin(); it != this->layers.rend(); it++) {
		err = (*it)->backprop(err, learning_rate);
	}

	cudaDeviceSynchronize();
}

std::vector<NNLayer*> NeuralNetwork::getLayers() const {
	return layers;
}