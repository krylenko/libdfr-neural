#ifndef DFRNEURALLINEARLAYER_H
#define DFRNEURALLINEARLAYER_H

#include "dfrNeuralLayer.h"

class NeuralLinearLayer: public NeuralLayer 
{

  public:
    NeuralLinearLayer(const vecIntType inputs, const vecIntType nodes);
    ~NeuralLinearLayer();
    
  protected:

};

#endif
