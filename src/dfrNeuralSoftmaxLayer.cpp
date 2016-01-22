// daniel ford 2015

#include <cmath>
#include <iostream>
#include "dfrNeuralLayer.h"
#include "dfrNeuralSoftmaxLayer.h"

NeuralSoftmaxLayer::NeuralSoftmaxLayer(const int inputs, const int nodes)
: NeuralLayer(inputs, nodes) 
{
  m_type = SOFTMAX;
}

NeuralSoftmaxLayer::~NeuralSoftmaxLayer()
{
}

std::vector<double> NeuralSoftmaxLayer::computeOutputs(const std::vector<double>& inputs)
{
  
  std::vector<double> outs(m_numNodes);
  outs = NeuralLayer::computeOutputs(inputs);

  double C = outs[0];
  for(auto i=0;i<outs.size();++i)
  {
    if( outs[i] > C ) C = outs[i];  
  }

  double normalizer = 0.0;
  for(std::vector<double>::iterator it=outs.begin();it!=outs.end();++it)
  {
    normalizer += exp(*it-C);
  }
  for(std::vector<double>::iterator it=outs.begin();it!=outs.end();++it)
  {  
    *it = exp(*it-C)/normalizer;
  }
  m_output = outs;
  return outs;
}

void NeuralSoftmaxLayer::updateWeights(const std::vector<double>& prevOut, const std::vector<double>& deltas, const double learningRate, const double momentum, const double decayRate)
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
      double deltaW = learningRate * ( deltas[j] * prev - ( decayRate * m_weights[i][j] ) );
      m_weights[i][j] -= deltaW + momentum * m_nextDeltas[i][j];
      m_nextDeltas[i][j] = deltaW;                  
    }
  }
}
