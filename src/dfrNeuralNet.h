// daniel ford 2015

#ifndef __DFRNEURALNET_H__
#define __DFRNEURALNET_H__

#include "dfrNeuralLayer.h"
#include "dfrNeuralLinearLayer.h"
#include "dfrNeuralTanhLayer.h"
#include "dfrNeuralSigmoidLayer.h"
#include "dfrNeuralSoftmaxLayer.h"
#include "dfrNeuralRectlinLayer.h"

// output enum to create correct training data
// use SCALAR for tanh/sigmoid output layers, 
// PROB for softmax output layers
enum{SCALAR,PROB};

class NeuralNet
{
  public:
    NeuralNet();
    ~NeuralNet();
  
    void addLayer(NeuralLayer * layer);
    
    double trainNet(const std::vector<double>& data, const std::vector<double>& labeledOutput, const unsigned int outType);
    
    std::vector<double> runNet(const std::vector<double>& data);
    
    int numLayers() { return m_layers.size(); }
    
    void setParams(const double& rate, const double& momentum, const double& decay, const unsigned int& outType);

    bool saveNet( const char * filename=nullptr );

    bool loadNet( const char * filename );
  
  private:

    std::vector<NeuralLayer *> m_layers;
    double m_learningRate;
    double m_momentum;
    double m_weightDecay;
    unsigned int m_outType;
    
    std::vector<double> minusVec(std::vector<double> one, std::vector<double> two);
    
    std::vector<double> computeOutput(const std::vector<double> & inputs);
    
    std::vector<double> computeError(const std::vector<double>& netOutput, const std::vector<double>& labeledOutput);

    double logloss(const std::vector<double>& netOutput, const std::vector<double>& labeledOutput);

};

#endif
