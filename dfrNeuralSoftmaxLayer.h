#ifndef DFRNEURALSOFTMAXLAYER_H
#define DFRNEURALSOFTMAXLAYER_H

#include "dfrNeuralLayer.h"

class NeuralSoftmaxLayer: public NeuralLayer
{
public:
    NeuralSoftmaxLayer(const vecIntType inputs, const vecIntType nodes);
    virtual ~NeuralSoftmaxLayer();
    
    virtual std::vector<double> computeOutputs(const std::vector<double>& inputs,
                                               const bool dropout);
    
    virtual void updateWeights(const std::vector<double>& prevOut, const std::vector<double>& deltas,
                               const double learningRate, const double momentum, const double decayRate);
};

#endif
