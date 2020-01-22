#ifndef DFRNEURALRELULAYER_H
#define DFRNEURALRELULAYER_H

#include "dfrNeuralLayer.h"

class NeuralReLULayer: public NeuralLayer
{
public:
    NeuralReLULayer(const vecIntType inputs, const vecIntType nodes);
    ~NeuralReLULayer();
    
protected:
    void activation();
};

#endif
