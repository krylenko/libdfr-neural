#include <cmath>
#include <iostream>
#include "dfrNeuralLayer.h"
#include "dfrNeuralTanhLayer.h"

NeuralTanhLayer::NeuralTanhLayer(const vecIntType inputs, const vecIntType nodes)
    : NeuralLayer(inputs, nodes)
{
    m_type = TANH;
}

NeuralTanhLayer::~NeuralTanhLayer()
{
}

void NeuralTanhLayer::activation()
{
    for (std::vector<double>::iterator it = m_output.begin(); it != m_output.end(); ++it) {
        *it = tanh(*it);
    }
}

double NeuralTanhLayer::outGrad(const double output)
{
    return 1.0 - (output * output);
}
