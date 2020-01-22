#include <cmath>
#include <iostream>
#include "dfrNeuralLayer.h"
#include "dfrNeuralSigmoidLayer.h"

NeuralSigmoidLayer::NeuralSigmoidLayer(const vecIntType inputs, const vecIntType nodes)
    : NeuralLayer(inputs, nodes)
{
    m_type = SIGMOID;
}

NeuralSigmoidLayer::~NeuralSigmoidLayer()
{
}

void NeuralSigmoidLayer::activation()
{
    for (std::vector<double>::iterator it = m_output.begin(); it != m_output.end(); ++it) {
        *it = 1.0 / (1.0 + exp(-1.0 * (*it)));
    }
}

double NeuralSigmoidLayer::outGrad(const double output)
{
    return output * (1.0 - output);
}
