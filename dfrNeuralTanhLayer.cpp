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

std::vector<double> NeuralTanhLayer::computeOutputs(const std::vector<double>& inputs,
                                                    const bool dropout)
{
    std::vector<double> outs(m_numNodes);
    outs = NeuralLayer::computeOutputs(inputs, dropout);
    for (std::vector<double>::iterator it=outs.begin(); it!=outs.end(); ++it) {
        *it = tanh(*it);
    }
    m_output = outs;
    return outs;
}

void NeuralTanhLayer::updateWeights(const std::vector<double>& prevOut, const std::vector<double>& deltas,
                                    const double learningRate, const double momentum, const double decayRate)
{
    for (vecIntType i = 0; i < m_inputEndIdx; ++i) {
        for (vecIntType j = 0; j < m_numNodes; ++j) {
            double prev = (i == 0) ? 1.0 : prevOut[i-1];
            double deltaW = learningRate * (deltas[j] * (1.0 - (m_output[j] * m_output[j])) *
                                            prev - (decayRate * m_weights[i][j]));
            m_weights[i][j] += deltaW + momentum * m_nextDeltas[i][j];
            m_nextDeltas[i][j] = deltaW;
        }
    }
}
