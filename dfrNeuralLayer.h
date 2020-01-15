#ifndef DFRNEURALLAYER_H
#define DFRNEURALLAYER_H

#include <vector>

// layer type enum
enum{LINEAR,TANH,SIGMOID,SOFTMAX};

enum{FALSE,TRUE};

using vecIntType = std::vector<int>::size_type;
using vecDblType = std::vector<double>::size_type;

class NeuralLayer
{
public:
  
    NeuralLayer(const vecIntType inputs, const vecIntType nodes, const int randSeed);
    virtual ~NeuralLayer();
    
    virtual vecIntType numInputs() {return m_numInputs;}

    virtual vecIntType numNodes() {return m_numNodes;}

    virtual std::vector<double> computeOutputs(const std::vector<double>& inputs, const bool dropout);

    virtual std::vector<double> retrieveOutputs() {return m_output;}

    virtual std::vector<std::vector<double> > retrieveWeights() {return m_weights;}

    virtual std::vector<double> computeDeltas(const std::vector<double>& error, const std::vector<std::vector<double> >& nextWeights);
    
    virtual void updateWeights(const std::vector<double>& prevOut, const std::vector<double>& deltas, const double learningRate, const double momentum, const double decayRate);
    
    virtual void loadWeights(const std::vector<std::vector<double> > newWeights) {m_weights=newWeights;}

    virtual vecIntType getType() {return m_type;}

protected:
  
    vecIntType m_type;
    vecIntType m_numInputs;
    vecIntType m_numNodes;
    std::vector<std::vector<double> > m_weights;
    std::vector<double> m_biases;
    std::vector<double> m_output;
    std::vector<std::vector<double> > m_nextDeltas;
  
};

#endif
