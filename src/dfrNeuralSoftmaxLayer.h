// daniel ford 2015

#ifndef __DFRNEURALSOFTMAXLAYER_H__
#define __DFRNEURALSOFTMAXLAYER_H__

#include "dfrNeuralLayer.h"

class NeuralSoftmaxLayer: public NeuralLayer
{

  public:
    NeuralSoftmaxLayer(const int inputs, const int nodes);
    virtual ~NeuralSoftmaxLayer();
    
    virtual std::vector<double> computeOutputs(const std::vector<double>& inputs);
    
    virtual void updateWeights(const std::vector<double>& prevOut, const std::vector<double>& deltas, const double learningRate, const double momentum, const double decayRate);
    
  protected:

};

#endif