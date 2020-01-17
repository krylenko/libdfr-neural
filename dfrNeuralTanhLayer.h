#ifndef DFRNEURALTANHLAYER_H
#define DFRNEURALTANHLAYER_H

#include "dfrNeuralLayer.h"

class NeuralTanhLayer: public NeuralLayer
{
public:
    NeuralTanhLayer(const vecIntType inputs, const vecIntType nodes);
    virtual ~NeuralTanhLayer();
    
    virtual std::vector<double> computeOutputs(const std::vector<double>& inputs,
                                               const bool training, const double dropoutRate);
    
    virtual void updateWeights(const std::vector<double>& prevOut, const std::vector<double>& deltas,
                               const double learningRate, const double momentum, const double decayRate);
};

#endif
