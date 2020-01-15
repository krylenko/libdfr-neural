#include "dfrNeuralLayer.h"
#include "dfrNeuralLinearLayer.h"

NeuralLinearLayer::NeuralLinearLayer(const vecIntType inputs, const vecIntType nodes,
                                     const int randSeed)
: NeuralLayer(inputs, nodes, randSeed)
{
    m_type = LINEAR;
}

NeuralLinearLayer::~NeuralLinearLayer()
{
}
