// daniel ford 2015

#ifndef __DFRNEURALLAYER_H__
#define __DFRNEURALLAYER_H__

#include <vector>

enum{FALSE,TRUE};

class NeuralLayer
{

  public:
  
    NeuralLayer(const int inputs, const int nodes);
    virtual ~NeuralLayer();
    
    virtual std::vector<double> computeOutputs(const std::vector<double>& inputs);

    virtual std::vector<double> retrieveOutputs() {return m_output;}

    virtual std::vector<std::vector<double> > retrieveWeights() {return m_weights;}

    virtual std::vector<double> computeDeltas(const std::vector<double>& error, const std::vector<std::vector<double> >& nextWeights);
    
    virtual void updateWeights(const std::vector<double>& prevOut, const std::vector<double>& deltas, const double learningRate, const double momentum, const double decayRate);
    
  protected:
  
    int m_numInputs;
    int m_numNodes;
    std::vector<std::vector<double> > m_weights;
	  std::vector<double> m_biases;
    std::vector<double> m_output;
    std::vector<std::vector<double> > m_nextDeltas;
  
};

#endif