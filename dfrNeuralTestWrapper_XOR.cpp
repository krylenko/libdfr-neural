// neural net example script: learning XOR

#include <iostream>
#include <cmath>
#include <cstdlib>
#include "dfrNeuralNet.h"

std::vector<double> XOR_training(const bool softmax);

int main(int, char**)
{ 
    const unsigned iters = 5000;
    const unsigned testers = 2500;

    // create network
    NeuralNet network;
    network.addLayer(std::shared_ptr<NeuralLayer>(new NeuralTanhLayer(2, 12)));
    network.addLayer(std::shared_ptr<NeuralLayer>(new NeuralSoftmaxLayer(12, 2)));

    // set learning rate, momentum, decay rate, dropout rate
    network.init(0.01, 0, 0, 1.0);

    // training
    for (unsigned i = 0; i < iters; ++i) {
        double error = 0.0;
        std::vector<double> exor, training;

        // generate training data
        bool softmax = false;
        switch(network.outType())
        {
        case SCALAR:
            exor = XOR_training(softmax);
            training.push_back(exor[2]);
            exor.pop_back();
            break;
        case PROB:
            softmax = true;
            exor = XOR_training(softmax);
            training.push_back(exor[2]);
            training.push_back(exor[3]);
            exor.pop_back();
            exor.pop_back();
            break;
        }
        auto netOut = network.trainNet(exor, training);
        error = netOut.first;
        /*
        auto outputs = netOut.second;
        switch(network.outType())
        {
        case SCALAR:
          //std::cout << exor[0] << " ^ " << exor[1] << " = " << outputs[0] << ", " << error << std::endl;
          break;
        case PROB:
          //std::cout << exor[0] << " ^ " << exor[1] << " = " << outputs[0] << " | " << outputs[1] << ", " << error << std::endl;
          break;
        }
        */
    }

    // testing
    int rightCount = 0;
    for (unsigned i = 0; i < testers; ++i) {

        bool softmax = network.outType() == SCALAR ? false : true;
        std::vector<double> exor = XOR_training(softmax);

        unsigned out, correctLabel;
        std::vector<double> outputs;
        switch(network.outType())
        {
        case SCALAR:
            correctLabel = unsigned(exor[2]);
            exor.pop_back();
            outputs = network.runNet(exor);
            out = ((outputs[0] > 0.5) ? 1 : 0);
            break;
        case PROB:
            correctLabel = unsigned(exor[2] > exor[3] ? 0 : 1);
            exor.pop_back();
            exor.pop_back();
            outputs = network.runNet(exor);
            out = (outputs[0] > outputs[1]) ? 0 : 1;
            break;
        }
        //std::cout << exor[0] << " ^ " << exor[1] << " = " << correctLabel << " , " << out << std::endl;
        if (out == correctLabel) {
            ++rightCount;
        }
    }
    std::cout << std::endl << "accuracy: " << 100.0 * rightCount / testers << "%" << std::endl;
    return 0;
}

//auto random XOR generator function
std::vector<double> XOR_training(const bool softmax)
{
    double x = 0.0;
    std::vector<double> exor;

    x = 1.0 * rand() / double(RAND_MAX);

    if (x <= 0.25) {
        exor.push_back(1.0); exor.push_back(0.0);
        if (softmax) {
            exor.push_back(0.0);
        }
        exor.push_back(1.0);
    }
    if ((x > 0.25) && (x <= 0.5)) {
        exor.push_back(0.0); exor.push_back(0.0);
        if (softmax) {
            exor.push_back(1.0);
        }
        exor.push_back(0.0);
    }
    if ((x > 0.5) && (x <= 0.75)) {
        exor.push_back(1.0); exor.push_back(1.0);
        if (softmax) {
            exor.push_back(1.0);
        }
        exor.push_back(0.0);
    }
    if ((x > 0.75) && (x <= 1.0)) {
        exor.push_back(0); exor.push_back(1.0);
        if (softmax) {
            exor.push_back(0.0);
        }
        exor.push_back(1.0);
    }
    return exor;
}
