// neural net example script: learning XOR

#include <iostream>
#include <math.h>
#include <stdlib.h>
#include "dfrNeuralNet.h"

std::vector<double> XOR_training(const bool softmax);

int main(int, char**)
{ 
    const unsigned iters = 5000;
    const unsigned testers = 2500;
    int seed = int(time(nullptr));

    // create network
    NeuralNet network;

    NeuralLayer * pHiddenLayer1 = new NeuralTanhLayer(2, 12, seed);
    network.addLayer(pHiddenLayer1);

    NeuralLayer * pOutputLayer = new NeuralSoftmaxLayer(12, 2, seed);
    //NeuralLayer * pOutputLayer = new NeuralLinearLayer(48, 1, seed);
    network.addLayer(pOutputLayer);

    // set learning rate, momentum, decay rate, output type
    // SCALAR = tanh or sigmoid output layer (use one output neuron)
    // PROB = softmax output layer, 1-of-C output encoding (use two output neurons)
    const unsigned int outType = PROB;
    //const unsigned int outType = SCALAR;
    network.setParams(0.01, 0, 0, outType);

    int rightCount = 0;

    for (unsigned i=0; i<iters+testers; ++i) {
        double error = 0.0;
        std::vector<double> exor, training;

        // generate training data
        bool softmax = false;
        switch(outType)
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

        // training
        if (i<iters){
            std::vector<double> outputs = network.runNet(exor);
            error = network.trainNet(exor, training, outType);
            switch(outType)
            {
            case SCALAR:
              //std::cout << exor[0] << " ^ " << exor[1] << " = " << outputs[0] << ", " << error << std::endl;
              break;
            case PROB:
              //std::cout << exor[0] << " ^ " << exor[1] << " = " << outputs[0] << " | " << outputs[1] << ", " << error << std::endl;
              break;
            }
        }

        // testing
        if (i >= iters) {
            std::vector<double> outputs = network.runNet(exor);
            unsigned out = 0;
            switch(outType)
            {
            case SCALAR:
                out = ((outputs[0] > 0.5) ? 1 : 0);
                if (out == unsigned(training[0])) {
                    ++rightCount;
                  }
                break;
            case PROB:
                vecIntType classLabel = 0;
                if (outputs[0] < outputs[1]) {
                    classLabel = 1;
                }
                if (1 == vecIntType(training[classLabel])) {
                    ++rightCount;
                }
                break;
            }
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
