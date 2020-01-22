#include <cmath>
#include <iostream>
#include "dfrNeuralLayer.h"
#include "dfrNeuralReLULayer.h"

NeuralReLULayer::NeuralReLULayer(const vecIntType inputs, const vecIntType nodes)
    : NeuralLayer(inputs, nodes)
{
    m_type = RELU;
}

NeuralReLULayer::~NeuralReLULayer()
{
}

void NeuralReLULayer::activation()
{
    for (std::vector<double>::iterator it = m_output.begin(); it != m_output.end(); ++it) {
        *it = std::max(0.0, *it);
    }
}
