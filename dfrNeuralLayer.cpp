#include "dfrNeuralLayer.h"
#include <iostream>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>

NeuralLayer::NeuralLayer(const vecIntType inputs, const vecIntType nodes, const int randSeed)
: m_numInputs(inputs)
, m_numNodes(nodes)
, m_weights(m_numInputs+1,std::vector<double>(m_numNodes, 0.0))
, m_biases(m_numNodes, 0.0)
, m_output(m_numNodes, 0.0)
, m_nextDeltas(m_numInputs+1,std::vector<double>(m_numNodes, 0.0))
{
    if (randSeed < 0) {
        srand(unsigned(time(nullptr)));
    } else {
        srand(unsigned(randSeed));
    }
    double r = 1.0 / sqrt(double(m_numInputs));

    for (vecIntType j=0; j<m_numNodes; ++j) {
        m_biases[j] = 1.0;
        for (vecIntType i=0; i<m_numInputs+1; ++i) {
            m_weights[i][j] = (2.0 * r * rand()/double(RAND_MAX)) - r;
        }
    }
}

NeuralLayer::~NeuralLayer()
{
}

std::vector<double> NeuralLayer::computeOutputs(const std::vector<double>& inputs,
                                                const bool dropout)
{
    assert( m_numInputs == inputs.size() );
    std::vector<double> outputs(m_numNodes, 0.0);

    for (vecIntType j=0; j<m_numNodes; ++j) {
        outputs[j] = m_biases[j]*m_weights[0][j];
        for (vecIntType i=1; i<m_numInputs+1; ++i) {
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

    std::vector<double> deltas(m_numNodes+1);
    for (vecIntType j=0; j<m_numNodes+1; ++j) {
        for (vecIntType i=0; i<nextLayerOuts; ++i) {
            deltas[j] += (first ? (error[i] * nextWeights[j][i]) : (error[i+1] * nextWeights[j][i]));
        }
    }
    first = 0;
    return deltas;
}

void NeuralLayer::updateWeights(const std::vector<double>& prevOut, const std::vector<double>& deltas, const double learningRate, const double momentum, const double decayRate)
{
    for (vecIntType i=0; i<m_numInputs+1; ++i) {
        for (vecIntType j=0; j<m_numNodes; ++j) {
            double prev = (i==0) ? 1.0 : prevOut[i-1];
            double deltaW = learningRate * (deltas[j] * prev - decayRate * m_weights[i][j]);
            m_weights[i][j] += deltaW + momentum * m_nextDeltas[i][j];
            m_nextDeltas[i][j] = deltaW;
        }
    }
}