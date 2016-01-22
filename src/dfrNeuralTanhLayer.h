// daniel ford 2015

#ifndef __DFRNEURALTANHLAYER_H__
#define __DFRNEURALTANHLAYER_H__

#include "dfrNeuralLayer.h"

class NeuralTanhLayer: public NeuralLayer
{

  public:
    NeuralTanhLayer(const int inputs, const int nodes);
    virtual ~NeuralTanhLayer();
    
    virtual std::vector<double> computeOutputs(const std::vector<double>& inputs);
    
    virtual void updateWeights(const std::vector<double>& prevOut, const std::vector<double>& deltas, const double learningRate, const double momentum, const double decayRate);
    
  protected:

};

#endif