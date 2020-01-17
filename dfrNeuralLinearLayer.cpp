#include "dfrNeuralLayer.h"
#include "dfrNeuralLinearLayer.h"

NeuralLinearLayer::NeuralLinearLayer(const vecIntType inputs, const vecIntType nodes)
    : NeuralLayer(inputs, nodes)
{
    m_type = LINEAR;
}

NeuralLinearLayer::~NeuralLinearLayer()
{
}
