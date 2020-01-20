#include "dfrNeuralLayer.h"

#include <iostream>
#include <numeric>
#include <cassert>
#include <ctime>
#include <cmath>
#include <cstdlib>

NeuralLayer::NeuralLayer(const vecIntType inputs, const vecIntType nodes)
    : m_numInputs(inputs)
    , m_numNodes(nodes)
    , m_weights(m_numInputs + 1, std::vector<double>(m_numNodes, 0.0))
    , m_biases(m_numNodes, 0.0)
    , m_output(m_numNodes, 0.0)
    , m_nextDeltas(m_numInputs + 1, std::vector<double>(m_numNodes, 0.0))
{
}

void NeuralLayer::initLayer(const unsigned weightInitType)
{
    double r = 0.0, initWeight = 0.0;
    for (vecIntType j = 0; j < m_numNodes; ++j) {
        m_biases[j] = 1.0;
        for (vecIntType i = 0; i < m_numInputs + 1; ++i) {
            double uniform = 2.0 * rand()/double(RAND_MAX);
            switch(weightInitType)
            {
            case SQRT:
                r = 1.0 / sqrt(double(m_numInputs));
                initWeight = (r * uniform) - r;
                break;
            case TRUNC_NORM:
                r = std::max(-1.0, uniform - 1.0);
                initWeight = std::min(1.0, r);
                break;
            }
            m_weights[i][j] = initWeight;
        }
    }
}

std::vector<double> NeuralLayer::computeOutputs(const std::vector<double>& inputs,
                                                const bool training, const double dropoutRate)
{
    assert(m_numInputs == inputs.size());
    std::vector<double> outputs(m_numNodes, 0.0);
    for (vecIntType j = 0; j < m_numNodes; ++j) {
        outputs[j] = m_biases[j] * m_weights[0][j];
        for (vecIntType i = 1; i < m_numInputs + 1; ++i) {
            outputs[j] += inputs[i-1] * m_weights[i][j];
        }
    }
    // apply dropout and scale outputs to adjust magnitudes for dropped-out nodes
    if (training && (dropoutRate < 1.0)) {
        for (vecIntType k = 0; k < m_numNodes; ++k) {
            double dropoutRand = rand() / double(RAND_MAX);
                if (dropoutRand >= dropoutRate) {
                    outputs[k] /= dropoutRate;
                } else {
                    outputs[k] = 0.0;
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

    std::vector<double> deltas(m_numNodes + 1);
    for (vecIntType j = 0; j < m_numNodes + 1; ++j) {
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
