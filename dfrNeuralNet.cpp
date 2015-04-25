// daniel ford 2015

#include "dfrNeuralLayer.h"
#include "dfrNeuralNet.h"
#include <assert.h>
#include <iostream>

NeuralNet::NeuralNet()
: m_layers(0)
, m_learningRate(0.0)
, m_momentum(0.0)
, m_weightDecay(0.0)
{
}

NeuralNet::~NeuralNet()
{
  for(std::vector<NeuralLayer *>::iterator it=m_layers.begin();it!=m_layers.end();++it)
  {
    delete *it;
  }
}

void NeuralNet::addLayer(NeuralLayer * layer)
{
  m_layers.push_back(layer);
}

void NeuralNet::setParams(const double& rate, const double& momentum, const double& decay  )
{
  m_learningRate = rate;
  m_weightDecay = decay;
  m_momentum = momentum;
}

std::vector<double> NeuralNet::minusVec(std::vector<double> one, std::vector<double> two)
{
  assert( one.size()==two.size() );
  std::vector<double> result;
	for(unsigned int i=0;i<one.size();++i)
	{
		result.push_back(one[i]-two[i]);
	}
	return result;
}

double computeMSE( const std::vector<double>& error )
{
  double MSE = 0.0;
  for(unsigned int i=0;i<error.size();++i)
  {
    MSE += error[i]*error[i];
  }
  return 0.5*MSE;
}

double NeuralNet::trainNet(const std::vector<double>& data, const std::vector<double>& trainingOutput, const unsigned int outType)
{

  int outputLayer = m_layers.size();
  int inputLayer = 0;
  
  std::vector<double> output,error,delta,prevOut,nextDeltas;
  double cost = 0.0;
  
  // run net forward
  output = runNet( data );
  error = computeError( output, trainingOutput );
  switch(outType)
  {
    case SCALAR:
      cost = computeMSE( error );
      break;
    case PROB:
      cost = logloss( output, trainingOutput );
      break;
  }

  // propagate error backward through layers
  for(int i=outputLayer; i>0; i--)
  {

    // calculate initial gradient
    if( i==outputLayer )
      delta = error;
    else
      delta = m_layers[i-1]->computeDeltas(error,m_layers[i]->retrieveWeights());
    if( i > 1)
      prevOut = m_layers[ i-1-1 ]->retrieveOutputs();
    if( i <= 1)
      prevOut = data;
    m_layers[i-1]->updateWeights( prevOut,delta,m_learningRate,m_momentum,m_weightDecay );
    error = delta;
  
  }
  
  return cost;

}

std::vector<double> NeuralNet::runNet(const std::vector<double>& data)
{
  return computeOutput(data);
}

std::vector<double> NeuralNet::computeOutput(const std::vector<double> & inputs)
{
  std::vector<double> ins(inputs);
  std::vector<double> outs;
  for(std::vector<NeuralLayer *>::iterator it=m_layers.begin();it!=m_layers.end();++it)
  {
    outs=(*it)->computeOutputs(ins);
    ins=outs;
  }
  return outs;
}

std::vector<double> NeuralNet::computeError(const std::vector<double>& netOutput, const std::vector<double>& labeledOutput)
{
  return minusVec(labeledOutput,netOutput);
}

double NeuralNet::logloss(const std::vector<double>& netOutput, const std::vector<double>& labeledOutput)
{
  double logloss = 0.0;

  for(unsigned int i=0;i<netOutput.size();++i)
  {
    logloss -= log(netOutput[i])*labeledOutput[i]; 
  }

  return logloss;
}
