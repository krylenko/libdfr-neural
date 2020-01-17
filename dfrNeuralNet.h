#ifndef DFRNEURALNET_H
#define DFRNEURALNET_H

#include "dfrNeuralLayer.h"
#include "dfrNeuralLinearLayer.h"
#include "dfrNeuralTanhLayer.h"
#include "dfrNeuralSigmoidLayer.h"
#include "dfrNeuralSoftmaxLayer.h"

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
    
    double trainNet(const std::vector<double>& data, const std::vector<double>& labeledOutput, const bool dropout=false);
    
    std::vector<double> runNet(const std::vector<double>& data, const bool dropout=false);
    
    vecIntType numLayers() { return m_layers.size(); }
    
    void init(const double rate, const double momentum, const double decay,
              const bool useBias=true, const unsigned weightInitType=SQRT,
              const int randSeed=int(time(nullptr)));

    bool saveNet(const char * filename=nullptr);

    bool loadNet(const char * filename);
  
    unsigned outType() {return m_outType;}

private:

    std::vector<NeuralLayer *> m_layers;
    double m_learningRate;
    double m_momentum;
    double m_weightDecay;
    unsigned m_outType;
    
    std::vector<double> minusVec(std::vector<double> one, std::vector<double> two);
    
    std::vector<double> computeOutput(const std::vector<double> & inputs, const bool dropout);
    
    std::vector<double> computeError(const std::vector<double>& netOutput, const std::vector<double>& labeledOutput);

    double logloss(const std::vector<double>& netOutput, const std::vector<double>& labeledOutput);

};

#endif
