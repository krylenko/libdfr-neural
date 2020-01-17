#ifndef DFRNEURALSIGMOIDLAYER_H
#define DFRNEURALSIGMOIDLAYER_H

#include "dfrNeuralLayer.h"

class NeuralSigmoidLayer: public NeuralLayer
{
public:
    NeuralSigmoidLayer(const vecIntType inputs, const vecIntType nodes);
    virtual ~NeuralSigmoidLayer();
    
    virtual std::vector<double> computeOutputs(const std::vector<double>& inputs,
                                               const bool dropout);
    
    virtual void updateWeights(const std::vector<double>& prevOut, const std::vector<double>& deltas,
                               const double learningRate, const double momentum, const double decayRate);
};

#endif
