// daniel ford 2015

#ifndef __DFRNEURALRECTLINLAYER_H__
#define __DFRNEURALRECTLINLAYER_H__

#include "dfrNeuralLayer.h"

class NeuralRectlinLayer: public NeuralLayer
{

  public:
    NeuralRectlinLayer(const int inputs, const int nodes);
    virtual ~NeuralRectlinLayer();
    
    virtual std::vector<double> computeOutputs(const std::vector<double>& inputs);
    
    virtual void updateWeights(const std::vector<double>& prevOut, const std::vector<double>& deltas, const double learningRate, const double momentum, const double decayRate);
    
  protected:

};

#endif
