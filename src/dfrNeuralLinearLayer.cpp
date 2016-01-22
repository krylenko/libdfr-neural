// daniel ford 2015

#include "dfrNeuralLayer.h"
#include "dfrNeuralLinearLayer.h"

NeuralLinearLayer::NeuralLinearLayer(const int inputs, const int nodes)
: NeuralLayer(inputs, nodes) 
{
  m_type = LINEAR;
}

NeuralLinearLayer::~NeuralLinearLayer()
{
}