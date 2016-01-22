// daniel ford 2015

#include "dfrNeuralNet.h"
#include <assert.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <sstream>
#include <cmath>

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

void NeuralNet::setParams(const double& rate, const double& momentum, const double& decay)
{
  m_learningRate = rate;
  m_weightDecay = decay;
  m_momentum = momentum;
}

double NeuralNet::trainNet(const std::vector<double>& data, const std::vector<double>& trainingOutput)
{

  int outputLayer = m_layers.size();
  
  std::vector<double> output,error,delta,prevOut,nextDeltas;
  double cost = 0.0;
  
  // run net forward
  output = runNet( data );
  error = m_layers[outputLayer-1]->computeError( output, trainingOutput );
  cost = m_layers[outputLayer-1]->computeCost( error );

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
    //for(auto m=0; m<delta.size(); ++m)    
    //    std::cout << " > D > " << delta[m] << " >> ";
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

double NeuralNet::logloss(const std::vector<double>& netOutput, const std::vector<double>& labeledOutput)
{
  double logloss = 0.0;

  for(unsigned int i=0;i<netOutput.size();++i)
  {
    logloss += ( labeledOutput[i]*log(netOutput[i]) + (1.0-labeledOutput[i])*log(1.0-netOutput[i]) );
    //logloss -= log(netOutput[i])*labeledOutput[i]; 
  }

  return -logloss;
}

// save network
bool NeuralNet::saveNet( const char * filename )
{

  unsigned int nLayers = numLayers();

  // don't save empty network
  if( nLayers == 0 )
  {
    return false;
  }

  std::ofstream outp;
  assert(outp);

  std::string temp;
  if( !filename )
  {
    time_t now = time(0);
    struct tm* localnow = localtime(&now);
    std::ostringstream fname;
    fname << "../data/";
    fname << "netsave_" << localnow->tm_mon+1 << "-" << localnow->tm_mday << "-" << localnow->tm_year+1900 << "_";
    fname << localnow->tm_hour << "-" << localnow->tm_min << "-" << localnow->tm_sec;
    fname << ".data";
    temp = fname.str().c_str();
  }
  else
    temp = filename;

  outp.open(temp, std::ios::out);

  // save info about network
  outp << nLayers << "\t";
  outp << m_learningRate << "\t";
  outp << m_momentum << "\t";
  outp << m_weightDecay << "\t";
  outp << std::endl << std::endl;

  // save individual layers
  for(unsigned int m=0;m<nLayers;++m)
  {
    outp << m_layers[m]->getType() << "\t";
    outp << m_layers[m]->numInputs() << "\t";
    outp << m_layers[m]->numNodes() << std::endl;
    auto weights = m_layers[m]->retrieveWeights();
    for(auto i=weights.begin();i<weights.end();++i)
    {
      for(auto j=i->begin();j<(i->end());++j)
      {
        outp << (*j) << "\t\t";
      }
      outp << std::endl;
    }
    outp << std::endl;
  }

  outp.close();

  return true;

}

bool NeuralNet::loadNet( const char * filename )
{
  // check for existing layers
  if( numLayers() != 0 )
  {
    return false;
  }

  std::ifstream inp;
  assert(inp);
  inp.open(filename, std::ios::in);

  // load network parameters
  unsigned int loadLayers = 0;
  double learnRate = 0.0;
  double mom = 0.0;
  double decay = 0.0;
  inp >> loadLayers;
  inp >> learnRate;
  inp >> mom;
  inp >> decay;
  setParams(learnRate,mom,decay);

  // construct network
  for(unsigned int m=0;m<loadLayers;++m)
  {
    unsigned int type = 0;
    inp >> type;
    unsigned int inputs = 0;
    inp >> inputs;
    unsigned int nodes = 0;
    inp >> nodes;
    std::vector<std::vector<double> > weights(inputs+1,std::vector<double>(nodes,0.0));
    for(unsigned int i=0;i<inputs+1;++i)
    {
      for(unsigned int j=0;j<nodes;++j)
      {
        inp >> weights[i][j];
      }
    }
    NeuralLayer * pHiddenLayer = nullptr;
    switch(type)
    {
      case(LINEAR):
      pHiddenLayer = new NeuralLinearLayer(inputs,nodes);
      break;
      case(TANH):
      pHiddenLayer = new NeuralTanhLayer(inputs,nodes);
      break;
      case(SIGMOID):
      pHiddenLayer = new NeuralSigmoidLayer(inputs,nodes);
      break;
      case(SOFTMAX):
      pHiddenLayer = new NeuralSoftmaxLayer(inputs,nodes);
      break;
      case(RECTLIN):
      pHiddenLayer = new NeuralRectlinLayer(inputs,nodes);
      break;
    }
    pHiddenLayer->loadWeights(weights);
    addLayer( pHiddenLayer );
  }

  inp.close();
  return true;

}