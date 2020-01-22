#ifndef DFRNEURALSIGMOIDLAYER_H
#define DFRNEURALSIGMOIDLAYER_H

#include "dfrNeuralLayer.h"

class NeuralSigmoidLayer: public NeuralLayer
{
public:
    NeuralSigmoidLayer(const vecIntType inputs, const vecIntType nodes);
    virtual ~NeuralSigmoidLayer();
    
protected:
    void activation();
    double outGrad(const double output);
};

#endif
