// daniel ford 2015

#include <cmath>
#include <iostream>
#include <algorithm>
#include "dfrNeuralLayer.h"
#include "dfrNeuralRectlinLayer.h"

NeuralRectlinLayer::NeuralRectlinLayer(const int inputs, const int nodes)
: NeuralLayer(inputs, nodes) 
{
  m_type = RECTLIN;
}

NeuralRectlinLayer::~NeuralRectlinLayer()
{
}

std::vector<double> NeuralRectlinLayer::computeOutputs(const std::vector<double>& inputs)
{
  std::vector<double> outs(m_numNodes);
  outs = NeuralLayer::computeOutputs(inputs);
  for(std::vector<double>::iterator it=outs.begin();it!=outs.end();++it)
  { 
    *it = std::max(0.0,*it);
  }
  m_output = outs;
  return outs;
}

void NeuralRectlinLayer::updateWeights(const std::vector<double>& prevOut, const std::vector<double>& deltas, const double learningRate, const double momentum, const double decayRate)
{
  for(int i=0; i<m_numInputs+1; ++i)
  {
    for(int j=0; j<m_numNodes; ++j)
    {
      double prev = 0.0;
      if(i==0)
        prev = 1.0;
      else
        prev = prevOut[i-1];

      double deltaW = 0.0;      
      if( m_output[j] >= 0.0 )
        deltaW = learningRate * ( deltas[j] * prev - decayRate * m_weights[i][j] ) ;    

      m_weights[i][j] -= deltaW + momentum * m_nextDeltas[i][j];
      m_nextDeltas[i][j] = deltaW;                  
    }
  }
}
