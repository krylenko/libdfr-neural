#include "dfrNeuralLayer.h"
#include <iostream>
#include <cassert>
#include <ctime>
#include <cmath>
#include <cstdlib>

NeuralLayer::NeuralLayer(const vecIntType inputs, const vecIntType nodes)
    : m_numInputs(inputs)
    , m_numNodes(nodes)
    , m_weights(m_numInputs, std::vector<double>(m_numNodes, 0.0))
    , m_output(m_numNodes, 0.0)
    , m_nextDeltas(m_numInputs, std::vector<double>(m_numNodes, 0.0))
{
}

void NeuralLayer::initLayer(const bool useBias, const unsigned weightInitType)
{
    m_useBias = useBias;
    m_inputEndIdx = m_useBias ? m_numInputs + 1 : m_numInputs;
    if (m_useBias) {
        auto emptyVec = std::vector<double>(m_numNodes, 0.0);
        m_biases = emptyVec;
        m_weights.push_back(emptyVec);
        m_nextDeltas.push_back(emptyVec);
    }
    double r = 1.0 / sqrt(double(m_numInputs));
    switch(weightInitType)
    {
    case SQRT:
        for (vecIntType j = 0; j < m_numNodes; ++j) {
            if (m_useBias) {
                m_biases[j] = 1.0;
            }
            for (vecIntType i = 0; i < m_inputEndIdx; ++i) {
                m_weights[i][j] = (2.0 * r * rand()/double(RAND_MAX)) - r;
            }
        }
        break;
    case TRUNC_NORM:
        break;
    }
}

std::vector<double> NeuralLayer::computeOutputs(const std::vector<double>& inputs,
                                                const bool dropout)
{
    assert( m_numInputs == inputs.size() );
    std::vector<double> outputs(m_numNodes, 0.0);

    vecIntType startIdx = 0;
    vecIntType endIdx = m_numInputs;
    if (m_useBias) {
        startIdx = 1;
        endIdx += 1;
    }

    for (vecIntType j = 0; j < m_numNodes; ++j) {
        if (m_useBias) {
            outputs[j] = m_biases[j] * m_weights[0][j];
        }
        for (vecIntType i = startIdx; i < endIdx; ++i) {
            double dropoutRand = rand() / double(RAND_MAX);
            if (!dropout || dropoutRand > 0.5) {
                outputs[j] += inputs[i-1] * m_weights[i][j];
            }
        }
    }
    m_output = outputs;
    return outputs;
}

std::vector<double> NeuralLayer::computeDeltas(const std::vector<double>& error, const std::vector<std::vector<double> >& nextWeights)
{
    static unsigned int first = 1;
    vecIntType nextLayerOuts = first ? error.size() : (error.size() - 1);

    vecIntType numNodes = m_numNodes;
    if (m_useBias) {
        numNodes += 1;
    }

    std::vector<double> deltas(numNodes);
    for (vecIntType j = 0; j < numNodes; ++j) {
        for (vecIntType i = 0; i < nextLayerOuts; ++i) {
            deltas[j] += (first ? (error[i] * nextWeights[j][i]) : (error[i+1] * nextWeights[j][i]));
        }
    }
    first = 0;
    return deltas;
}

void NeuralLayer::updateWeights(const std::vector<double>& prevOut, const std::vector<double>& deltas, const double learningRate, const double momentum, const double decayRate)
{
    for (vecIntType i = 0; i < m_numInputs + 1; ++i) {
        for (vecIntType j = 0; j < m_numNodes; ++j) {
            double prev = (i == 0) ? 1.0 : prevOut[i-1];
            double deltaW = learningRate * (deltas[j] * prev - decayRate * m_weights[i][j]);
            m_weights[i][j] += deltaW + momentum * m_nextDeltas[i][j];
            m_nextDeltas[i][j] = deltaW;
        }
    }
}
