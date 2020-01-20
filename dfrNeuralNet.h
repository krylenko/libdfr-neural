#ifndef DFRNEURALNET_H
#define DFRNEURALNET_H

#include <algorithm>
#include <memory>

#include "dfrNeuralLayer.h"
#include "dfrNeuralLinearLayer.h"
#include "dfrNeuralTanhLayer.h"
#include "dfrNeuralSigmoidLayer.h"
#include "dfrNeuralSoftmaxLayer.h"
#include "DataLoader.h"

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
    void init(const double learningRate, const double momentum, const double decayRate,
              const double dropoutRate=1.0, const int randSeed=int(time(nullptr)),
              const unsigned weightInitType=SQRT);
    
    std::pair<double, double> train(DataLoader* dataset, const unsigned long epochs, const bool shuffleData);
    double trainNet(const std::vector<double>& data, const std::vector<double>& labeledOutput);

    double test(DataLoader* dataset);
    std::vector<double> runNet(const std::vector<double>& data);
    
    bool saveNet(const char * filename=nullptr);
    bool loadNet(const char * filename);
  
    vecIntType numLayers() { return m_layers.size(); }
    unsigned outType() {return m_outType;}

private:

    std::vector<double> minusVec(const std::vector<double>& one, const std::vector<double>& two);
    
    std::vector<double> computeOutput(const std::vector<double>& inputs, const bool training=true);
    
    std::vector<double> computeError(const std::vector<double>& netOutput, const std::vector<double>& labeledOutput);

    double logLoss(const std::vector<double>& netOutput, const std::vector<double>& labeledOutput);

    // convert single input digit label into 1-of-N encoded matrix
    std::vector<double> encodeOneHot(const vecIntType digit);

    // convert 1-of-N float output matrix to single digit
    vecIntType decodeOneHot(std::vector<double>& netOut);

    std::vector<NeuralLayer *> m_layers;
    double m_learningRate;
    double m_momentum;
    double m_weightDecay;
    double m_dropoutRate;
    unsigned m_outType;

};

#endif
