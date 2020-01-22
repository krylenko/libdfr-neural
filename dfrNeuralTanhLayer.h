#ifndef DFRNEURALTANHLAYER_H
#define DFRNEURALTANHLAYER_H

#include "dfrNeuralLayer.h"

class NeuralTanhLayer: public NeuralLayer
{
public:
    NeuralTanhLayer(const vecIntType inputs, const vecIntType nodes);
    virtual ~NeuralTanhLayer();
    
protected:
    void activation();
    double outGrad(const double output);
};

#endif
