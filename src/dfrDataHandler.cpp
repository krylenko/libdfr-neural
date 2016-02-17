// daniel ford 2015

#include <iostream>
#include <fstream>
#include <set>
#include <time.h>

#include "dfrDataHandler.h"

// params:
// 0    program name, not needed
// 1    learning rate
// 2    momentum
// 3    decay rate
// 4    path to data
#define NUM_CMD_ARGS    5

DataHandler::DataHandler(int paramCount_, char ** params_, NeuralNet * net_, unsigned int inputs_, unsigned int outputs_, unsigned int numPts_, unsigned int task_)
: _net(net_),
  _task(task_),
  _inputs(inputs_),
  _numClasses(outputs_),
  _numPts(numPts_),
  _timeStart(0),
  _timeStop(0)
{

    if(paramCount_ < NUM_CMD_ARGS ){
        _printUsage();
        exit(-1);
    }

    _loadParams(params_);

    _allData = new DVEC( numPts_, std::vector<double>(inputs_+1,0.0) );
    _loadData(*_allData, params_[4]);   

    srand((int)time(NULL));
}

DataHandler::~DataHandler()
{
    delete _allData;
}

void DataHandler::printVec(const std::vector<double>& vec)
{
    std::cout << " > ";
    for(unsigned int i=0; i<vec.size(); ++i)
    {
        std::cout << vec[i] << " > ";
    }
    std::cout << std::endl;
}

bool DataHandler::_loadParams(char ** params_)
{

    if(!_net)
        return false;

    std::cout << std::endl << "initializing network... ";

    double learningRate = std::stod(params_[1]);
    double momentum = std::stod(params_[2]);
    double decayRate = std::stod(params_[3]);

    _net->setParams(learningRate,momentum,decayRate);

    std::cout << "done" << std::endl;

    // print useful info for reference
    std::cout << "hidden layers: " << _net->numLayers()-1 << std::endl;

    std::cout << "\n" << "training examples: " << _numPts << std::endl;
    std::cout << "learning rate: " << learningRate << std::endl;
    std::cout << "momentum: " << momentum << std::endl;
    std::cout << "weight decay: " << decayRate << std::endl << std::endl;

    return true;
}

bool DataHandler::_loadData(DVEC& data_, const char * filename_)
{

    std::cout << "loading data... ";

    std::ifstream inp;
    inp.open(filename_, std::ios::in);
    if(inp){
        for(unsigned int i=0; i<data_.size(); ++i){
            for(unsigned int j=0;j<data_[0].size();++j){
                inp >> data_[i][j];
            }
        }

        inp.close();
    
        std::cout << "done" << std::endl;

        return true;
    }
    else
        return false;
}

bool DataHandler::sliceData(DVEC& trainData_, DVEC& testData_, DVEC& validData_, unsigned int numTrain_, unsigned int numTest_)
{

    unsigned int rndIdx = 0;
    std::set<unsigned int> idxSet;

    unsigned int cnt = 0;
    for(unsigned int i=0; i<_numPts; ++i){

        // get a random index in the data that hasn't already been used
        rndIdx = _shuffle(_numPts);
        if( idxSet.find(rndIdx) == idxSet.end() )
            idxSet.insert(rndIdx);
        else {
            while( idxSet.find(rndIdx) == idxSet.end() )
                rndIdx = _shuffle(_numPts);
            idxSet.insert(rndIdx);
        }

        for(unsigned int j=0;j<_inputs+1;++j){

            if( cnt < numTrain_ ){
                if( j==0 )
                    trainData_[cnt][j] = (*_allData)[rndIdx][j];
                else
                    trainData_[cnt][j] = _norm((*_allData)[rndIdx][j]);
            }

            if( cnt >= numTrain_ && i < (numTrain_ + numTest_ ) ){
                if( j==0 )
                    testData_[cnt-numTrain_][j] = (*_allData)[rndIdx][j];
                else
                    testData_[cnt-numTrain_][j] = _norm((*_allData)[rndIdx][j]);
            }            

            if( cnt > (numTrain_ + numTest_ ) ){
                if( j==0 )
                    validData_[cnt-numTrain_-numTest_][j] = (*_allData)[rndIdx][j];
                else
                    validData_[cnt-numTrain_-numTest_][j] = _norm((*_allData)[rndIdx][j]);
            }

        }

        ++cnt;

    }
    return true;
}

bool DataHandler::_saveNet()
{
    return true;
}

bool DataHandler::_loadNet()
{
    return true;
}

// convert class index to 1-of-N float vector 
std::vector<double> DataHandler::classToOneHot(unsigned int class_)
{

  std::vector<double> output(_numClasses,-1.0);
  output[class_] = 1.;
  return output;

}

// convert 1-of-N float output vector to single digit indicating the class
unsigned int DataHandler::oneHotToClass(const std::vector<double>& oneHot_)
{

  unsigned int classIdx = 0; double tmp = oneHot_[0];
  for(unsigned int i = 0; i<oneHot_.size(); ++i)
  {
    if(oneHot_[i] > tmp)
    {
      classIdx = i;
      tmp = oneHot_[i];
    }
  }
  return classIdx;

}

void DataHandler::timeStart()
{   
    _timeStart = (int)time(NULL);
}

int DataHandler::timeStop()
{
    _timeStop = (int)time(NULL);
    return _timeStop - _timeStart;
}

// return random index for data of given size in range [0,size-1]
unsigned int DataHandler::_shuffle(const unsigned int& size)
{
  return (unsigned int)(size*(double)rand()/(double)RAND_MAX);
}

double DataHandler::_norm(double input)
{
    return input/255.0;
}

void DataHandler::_printUsage()
{
    std::cout << std::endl;
    std::cout << "usage is: mnist <learning_rate> <momentum> <decay_rate> <path_to_data>" << std::endl;
    std::cout << std::endl;
}

