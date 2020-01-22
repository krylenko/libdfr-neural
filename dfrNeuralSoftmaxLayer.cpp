#include <algorithm>
#include <cmath>
#include <iostream>

#include "dfrNeuralLayer.h"
#include "dfrNeuralSoftmaxLayer.h"

NeuralSoftmaxLayer::NeuralSoftmaxLayer(const vecIntType inputs, const vecIntType nodes)
    : NeuralLayer(inputs, nodes)
{
    m_type = SOFTMAX;
}

NeuralSoftmaxLayer::~NeuralSoftmaxLayer()
{
}

void NeuralSoftmaxLayer::activation()
{
    double normalizer = 0.0;
    std::vector<double> expd(m_output.size(), 0.0);

    // compute normalization denom, save exponentiated values for later
    for (unsigned long i = 0; i < m_output.size(); ++i) {
        auto thisExp = exp(m_output[i]);
        normalizer += thisExp;
        expd[i] = thisExp;
    }
    // reuse exponentiated values to update outputs
    for (unsigned long p = 0; p < m_output.size(); ++p) {
        m_output[p] = expd[p] / normalizer;
    }
}
