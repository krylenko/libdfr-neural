// daniel ford 2015

#include "dfrNeuralLayer.h"
#include <iostream>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>

NeuralLayer::NeuralLayer(const int inputs, const int nodes) 
: m_numInputs(inputs)
, m_numNodes(nodes)
, m_weights(m_numInputs+1,std::vector<double>(m_numNodes,0.0))
, m_biases(m_numNodes,0.0)
, m_output(m_numNodes,0.0)
, m_nextDeltas(m_numInputs+1,std::vector<double>(m_numNodes,0.0))
{

  srand(time(NULL));
  
  double r = 1.0/sqrt((double)m_numInputs);

  for(int j=0;j<m_numNodes;++j)
  {
    m_biases[j] = 1.0;
    for(int i=0;i<m_numInputs+1;++i)    
    {
      m_weights[i][j] = (2.0*r*rand()/(double)RAND_MAX)-r;
    }
  }
}

NeuralLayer::~NeuralLayer()
{
}

std::vector<double> NeuralLayer::computeOutputs(const std::vector<double>& inputs)
{
  assert( m_numInputs==inputs.size() );
  std::vector<double> outputs(m_numNodes,0.0);
  
  for(int j=0;j<m_numNodes;++j)
  {
    outputs[j] = m_biases[j]*m_weights[0][j];
    for(int i=1;i<m_numInputs+1;++i)
    {
      outputs[j] += inputs[i-1]*m_weights[i][j];
    }
  }
  
  m_output = outputs;
  return outputs;
  
}

std::vector<double> NeuralLayer::computeDeltas(const std::vector<double>& error, const std::vector<std::vector<double> >& nextWeights)
{
  static unsigned int first = 1;
  int nextLayerOuts;
  
  if(first)
    nextLayerOuts = error.size();
  else
    nextLayerOuts = error.size()-1;

  std::vector<double> deltas(m_numNodes+1);
  for(int j=0; j<m_numNodes+1; ++j)
  {
    for(int i=0; i<nextLayerOuts; ++i)
    {
      if(first)
        deltas[j] += error[i] * nextWeights[j][i];
      else
        deltas[j] += error[i+1] * nextWeights[j][i];
    }
  }
  first = 0;
  return deltas;
}

void NeuralLayer::updateWeights(const std::vector<double>& prevOut, const std::vector<double>& deltas, const double learningRate, const double momentum, const double decayRate)
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
      double deltaW = learningRate * ( deltas[j] * prev - decayRate * m_weights[i][j] );
      m_weights[i][j] += deltaW + momentum * m_nextDeltas[i][j];
      m_nextDeltas[i][j] = deltaW;
    }
  }
}