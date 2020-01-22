#ifndef DFRNEURALSOFTMAXLAYER_H
#define DFRNEURALSOFTMAXLAYER_H

#include "dfrNeuralLayer.h"

class NeuralSoftmaxLayer: public NeuralLayer
{
public:
    NeuralSoftmaxLayer(const vecIntType inputs, const vecIntType nodes);
    virtual ~NeuralSoftmaxLayer();

protected:
    void activation();
};

#endif
