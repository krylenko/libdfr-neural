// daniel ford 2015

#ifndef __DFRNEURALNET_H__
#define __DFRNEURALNET_H__

#include "dfrNeuralLayer.h"
#include "dfrNeuralLinearLayer.h"
#include "dfrNeuralTanhLayer.h"
#include "dfrNeuralSigmoidLayer.h"
#include "dfrNeuralSoftmaxLayer.h"
#include "dfrNeuralRectlinLayer.h"

class NeuralNet
{
  public:
    NeuralNet();
    ~NeuralNet();
  
    void addLayer(NeuralLayer * layer);
    
    double trainNet(const std::vector<double>& data, const std::vector<double>& labeledOutput);
    
    std::vector<double> runNet(const std::vector<double>& data);
    
    int numLayers() { return m_layers.size(); }
    
    void setParams(const double& rate, const double& momentum, const double& decay);

    bool saveNet( const char * filename=nullptr );

    bool loadNet( const char * filename );
  
  private:

    std::vector<NeuralLayer *> m_layers;
    double m_learningRate;
    double m_momentum;
    double m_weightDecay;
    
    std::vector<double> computeOutput(const std::vector<double> & inputs);

    double logloss(const std::vector<double>& netOutput, const std::vector<double>& labeledOutput);

};

#endif
