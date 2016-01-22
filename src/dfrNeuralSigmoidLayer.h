// daniel ford 2015

#ifndef __DFRNEURALSIGMOIDLAYER_H__
#define __DFRNEURALSIGMOIDLAYER_H__

#include "dfrNeuralLayer.h"

class NeuralSigmoidLayer: public NeuralLayer
{

  public:
    NeuralSigmoidLayer(const int inputs, const int nodes);
    virtual ~NeuralSigmoidLayer();
    
    virtual std::vector<double> computeOutputs(const std::vector<double>& inputs);
    
    virtual void updateWeights(const std::vector<double>& prevOut, const std::vector<double>& deltas, const double learningRate, const double momentum, const double decayRate);
    
  protected:

};

#endif