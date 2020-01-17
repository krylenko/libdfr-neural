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

std::vector<double> NeuralSigmoidLayer::computeOutputs(const std::vector<double>& inputs, const bool training,
                                                       const double dropoutRate)
{
    std::vector<double> outs(m_numNodes);
    outs = NeuralLayer::computeOutputs(inputs, training, dropoutRate);
    for (std::vector<double>::iterator it=outs.begin(); it!=outs.end(); ++it) {
        // logistic function
        *it = 1 / (1 + exp(-1.0 * (*it)));
    }
    m_output = outs;
    return outs;
}

void NeuralSigmoidLayer::updateWeights(const std::vector<double>& prevOut, const std::vector<double>& deltas,
                                       const double learningRate, const double momentum, const double decayRate)
{
    for (vecIntType i = 0; i < m_numInputs + 1; ++i) {
        for (vecIntType j = 0; j < m_numNodes; ++j) {
            double prev = (i == 0) ? 1.0 :prevOut[i-1];
            double deltaW = learningRate * (deltas[j] * (m_output[j] * (1.0 - m_output[j])) *
                                            prev - (decayRate * m_weights[i][j]));
            m_weights[i][j] += deltaW + momentum * m_nextDeltas[i][j];
            m_nextDeltas[i][j] = deltaW;
        }
    }
}
