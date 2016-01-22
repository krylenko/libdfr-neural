// daniel ford 2015

#ifndef __DFRNEURALLINEARLAYER_H__
#define __DFRNEURALLINEARLAYER_H__

// libdfr-neural refactor
// daniel ford 2015

#include "dfrNeuralLayer.h"

class NeuralLinearLayer: public NeuralLayer 
{

  public:
    NeuralLinearLayer(const int inputs, const int nodes);
    ~NeuralLinearLayer();
    
  protected:

};

#endif