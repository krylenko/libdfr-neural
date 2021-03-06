#include "dfrNeuralNet.h"

#include <cassert>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>

double boundZeroOne(const double in)
{
    return std::min(1.0, std::max(0.0, in));
}

NeuralNet::NeuralNet()
: m_layers(0)
, m_learningRate(0.0)
, m_momentum(0.0)
, m_weightDecay(0.0)
, m_outType(SCALAR)
{
}

NeuralNet::~NeuralNet()
{
}

void NeuralNet::addLayer(std::shared_ptr<NeuralLayer> layer)
{
    m_layers.push_back(layer);
}

void NeuralNet::init(const double learningRate, const double momentum, const double decayRate,
                     const double dropoutRate, const int randSeed, const unsigned weightInitType)
{
    m_learningRate = boundZeroOne(learningRate);
    m_weightDecay = boundZeroOne(decayRate);
    m_momentum = boundZeroOne(momentum);
    m_dropoutRate = boundZeroOne(dropoutRate);

    // seed random number generator
    srand(unsigned(randSeed));

    for (auto it = m_layers.begin(); it != m_layers.end(); ++it) {
        (*it)->initLayer(weightInitType);
        m_outType = (*it)->getType() == SOFTMAX ? PROB : SCALAR;
    }
}

std::vector<double> NeuralNet::minusVec(const std::vector<double>& one, const std::vector<double>& two)
{
    assert(one.size() == two.size());
    std::vector<double> result;
    for (unsigned int i = 0; i < one.size(); ++i) {
        result.push_back(one[i] - two[i]);
    }
    return result;
}

double computeMSE(const std::vector<double>& error)
{
    double MSE = 0.0;
    for (unsigned int i = 0; i < error.size(); ++i) {
        MSE += error[i] * error[i];
    }
    return 0.5 * MSE;
}

std::pair<double, double> NeuralNet::train(std::shared_ptr<DataLoader> dataset, const unsigned long epochs,
                                           const bool shuffleData)
{
    double error = 0.0;
    unsigned long trueCount = 0;
    for (unsigned long p = 0; p < epochs; ++p) {
        dataset->splitTrainTest(shuffleData);
        for (vecIntType i = 0; i < dataset->numTrainPoints(); ++i) {
            auto thisPt = dataset->trainDataPoint();
            auto label = thisPt.first[0];
            auto data = thisPt.second;
            std::vector<double> nLabel = encodeOneHot(vecIntType(label));
            auto netOut = trainNet(data, nLabel);
            error = netOut.first;

            // decode output and compare to correct output
            if (decodeOneHot(netOut.second) == vecIntType(label)) {
                trueCount++;
            }
        }
    }
    double accuracy = trueCount * 100. / (dataset->numTrainPoints() * epochs);
    return std::make_pair(accuracy, error);
}

std::pair<double, std::vector<double>> NeuralNet::trainNet(const std::vector<double>& data, const std::vector<double>& trainingOutput)
{ 
    vecIntType outputLayer = m_layers.size();

    std::vector<double> output, error, delta, prevOut, nextDeltas;
    double loss = 0.0;

    // run net forward
    output = runNet(data);
    error = computeError(output, trainingOutput);
    loss = (m_outType == SCALAR) ? computeMSE(error) : logLoss(output, trainingOutput);

    // propagate error backward through layers
    for (vecIntType i=outputLayer; i>0; i--) {
        // calculate initial gradient
        if (i == outputLayer) {
            delta = error;
        } else {
          delta = m_layers[i-1]->computeDeltas(error, m_layers[i]->retrieveWeights());
        }
        prevOut = (i > 1) ? m_layers[i-1-1]->retrieveOutputs() : data;
        m_layers[i-1]->updateWeights(prevOut, delta, m_learningRate, m_momentum, m_weightDecay);
        error = delta;
    }
    return std::make_pair(loss, output);
}

double NeuralNet::test(std::shared_ptr<DataLoader> dataset)
{
    unsigned long trueCount = 0;
    for (vecIntType i = 0; i < dataset->numTestPoints(); ++i) {
        auto testPt = dataset->testDataPoint();
        auto label = testPt.first[0];
        auto data = testPt.second;

        std::vector<double> outputs = runNet(data);    // run net

        // decode output and compare to correct output
        if (decodeOneHot(outputs) == vecIntType(label)) {
            trueCount++;
        }
    }
    return trueCount * 100. / dataset->numTestPoints();
}

double NeuralNet::test(std::shared_ptr<DataMap_t> data)
{
    unsigned long trueCount = 0;
    for (auto it = data->begin(); it != data->end(); ++it) {
        auto testPt = it->second;
        auto label = testPt.first[0];
        auto data = testPt.second;

        std::vector<double> outputs = runNet(data);    // run net

        // decode output and compare to correct output
        if (decodeOneHot(outputs) == vecIntType(label)) {
            trueCount++;
        }
    }
    return trueCount * 100. / data->size();
}

std::vector<double> NeuralNet::runNet(const std::vector<double>& data)
{
    const bool training = false;
    return computeOutput(data, training);
}

std::vector<double> NeuralNet::computeOutput(const std::vector<double>& inputs, const bool training)
{
    std::vector<double> ins(inputs);
    std::vector<double> outs;
    for (auto it = m_layers.begin(); it != m_layers.end(); ++it) {
        auto layer = *it;
        layer->computeOutputs(ins, training, m_dropoutRate);
        outs = layer->retrieveOutputs();
        ins = outs;
    }
    return outs;
}

std::vector<double> NeuralNet::computeError(const std::vector<double>& netOutput, const std::vector<double>& labeledOutput)
{
    return minusVec(labeledOutput, netOutput);
}

double NeuralNet::logLoss(const std::vector<double>& netOutput, const std::vector<double>& labeledOutput)
{
    double logloss = 0.0;
    for (unsigned int i = 0; i < netOutput.size(); ++i) {
        logloss -= log(netOutput[i])*labeledOutput[i];
    }
    return logloss;
}

// save network
bool NeuralNet::saveNet(const char * filename)
{ 
    vecIntType nLayers = numLayers();

    // don't save empty network
    if (nLayers == 0) {
        return false;
    }

    std::ofstream outp;
    assert(outp);

    std::string temp;
    if (!filename) {
        time_t now = time(nullptr);
        struct tm* localnow = localtime(&now);
        std::ostringstream fname;
        fname << "netsave_" << localnow->tm_mon + 1 << "-" << localnow->tm_mday << "-" << localnow->tm_year + 1900 << "_";
        fname << localnow->tm_hour << "-" << localnow->tm_min << "-" << localnow->tm_sec;
        fname << ".data";
        temp = fname.str().c_str();
    } else {
        temp = filename;
    }

    outp.open(temp, std::ios::out);

    // save info about network
    outp << nLayers << "\t";
    outp << m_learningRate << "\t";
    outp << m_momentum << "\t";
    outp << m_weightDecay << "\t";
    outp << m_dropoutRate << "\t";
    outp << m_outType << "\t";
    outp << std::endl << std::endl;

    // save individual layers
    for (unsigned int m=0; m<nLayers; ++m) {
        outp << m_layers[m]->getType() << "\t";
        outp << m_layers[m]->numInputs() << "\t";
        outp << m_layers[m]->numNodes() << std::endl;
        auto weights = m_layers[m]->retrieveWeights();
        for (auto i=weights.begin(); i<weights.end(); ++i) {
            for (auto j=i->begin(); j<(i->end()); ++j) {
                outp << (*j) << "\t\t";
            }
            outp << std::endl;
        }
        outp << std::endl;
    }
    outp.close();
    return true;
}

bool NeuralNet::loadNet(const char * filename)
{
    // check for existing layers
    if (numLayers() != 0) {
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
    double dropout = 0.0;
    inp >> loadLayers;
    inp >> learnRate;
    inp >> mom;
    inp >> decay;
    inp >> dropout;
    init(learnRate, mom, decay, dropout);

    // construct network
    for (unsigned int m=0; m<loadLayers; ++m) {
        unsigned int type = 0;
        inp >> type;
        vecIntType inputs = 0;
        inp >> inputs;
        vecIntType nodes = 0;
        inp >> nodes;
        std::vector<std::vector<double> > weights(inputs+1,std::vector<double>(nodes,0.0));
        for (vecIntType i=0; i<inputs+1; ++i) {
            for (vecIntType j=0; j<nodes; ++j) {
                inp >> weights[i][j];
            }
        }
        std::shared_ptr<NeuralLayer> pHiddenLayer;
        switch(type)
        {
        case(LINEAR):
            pHiddenLayer = std::shared_ptr<NeuralLayer>(new NeuralLinearLayer(inputs, nodes));
            break;
        case(TANH):
            pHiddenLayer = std::shared_ptr<NeuralLayer>(new NeuralTanhLayer(inputs, nodes));
            break;
        case(SIGMOID):
            pHiddenLayer = std::shared_ptr<NeuralLayer>(new NeuralSigmoidLayer(inputs, nodes));
            break;
        case(SOFTMAX):
            pHiddenLayer = std::shared_ptr<NeuralLayer>(new NeuralSoftmaxLayer(inputs, nodes));
            break;
        }
        pHiddenLayer->loadWeights(weights);
        addLayer(pHiddenLayer);
    }
    inp.close();
    return true;
}

// convert single input digit label into 1-of-N encoded matrix
std::vector<double> NeuralNet::encodeOneHot(const vecIntType digit)
{
    std::vector<double> output(m_layers[m_layers.size()-1]->numNodes(), 0.);
    output[digit] = 1.;
    return output;
}

// convert 1-of-N float output matrix to single digit
vecIntType NeuralNet::decodeOneHot(std::vector<double>& netOut)
{
    vecIntType digit = 0;
    double tmp = netOut[0];
    for (vecIntType i = 0; i < m_layers[m_layers.size()-1]->numNodes(); ++i) {
        if (netOut[i] > tmp) {
            digit = i;
            tmp = netOut[i];
        }
    }
    return digit;
}
