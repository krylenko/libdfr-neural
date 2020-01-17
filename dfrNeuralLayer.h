#ifndef DFRNEURALLAYER_H
#define DFRNEURALLAYER_H

#include <ctime>
#include <vector>

// layer type enum
enum{LINEAR, TANH, SIGMOID, SOFTMAX};

// weight initialization type enum
enum{SQRT, TRUNC_NORM};

enum{FALSE, TRUE};

using vecIntType = std::vector<int>::size_type;

class NeuralLayer
{
public:
  
    NeuralLayer(const vecIntType inputs, const vecIntType nodes);
    virtual ~NeuralLayer() {}
    
    virtual vecIntType numInputs() {return m_numInputs;}

    virtual vecIntType numNodes() {return m_numNodes;}

    virtual vecIntType getType() {return m_type;}

    virtual std::vector<double> retrieveOutputs() {return m_output;}

    virtual std::vector<std::vector<double> > retrieveWeights() {return m_weights;}

    virtual void loadWeights(const std::vector<std::vector<double> > newWeights) {m_weights = newWeights;}

    virtual void initLayer(const unsigned weightInitType);

    virtual std::vector<double> computeOutputs(const std::vector<double>& inputs, const bool training, const double dropoutRate);

    virtual std::vector<double> computeDeltas(const std::vector<double>& error,
                                              const std::vector<std::vector<double> >& nextWeights);
    
    virtual void updateWeights(const std::vector<double>& prevOut, const std::vector<double>& deltas,
                               const double learningRate, const double momentum, const double decayRate);

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
