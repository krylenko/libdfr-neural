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

DataHandler::DataHandler(int paramCount_, char ** params_, NeuralNet * net_,
                         unsigned int inputs_, unsigned int outputs_, unsigned int numPts_)
: _net(net_),
  _inputs(inputs_),
  _numClasses(outputs_),
  _numPts(numPts_),
  _numTrain((unsigned int)(_numPts*0.6)),
  _numTest(_numPts-_numTrain),
  _numValid(0),
  _timeStart(0),
  _timeStop(0),
  _trueCnt(0)
{

    if(paramCount_ < NUM_CMD_ARGS ){
        printUsage();
        exit(-1);
    }

    loadParams(params_);

    _allData = new DVEC( _numPts, std::vector<double>(_inputs+1,0.0) );
    _trainData = new DVEC( _numTrain, std::vector<double>(_inputs+1,0.0) );
    _testData = new DVEC( _numTest, std::vector<double>(_inputs+1,0.0) );
    _validData = new DVEC( _numValid, std::vector<double>(_inputs+1,0.0) );
    loadData(_allData, params_[4]);

    srand((int)time(NULL));
}

DataHandler::~DataHandler()
{
    delete _allData;
}

void DataHandler::trainNet(unsigned int epochs_)
{

    std::cout << "\n" << "training points: " << _numTrain << std::endl;
    std::cout << "training network... " << std::endl;
    timeStart();

    for(unsigned int m=0; m<epochs_; ++m){

        double error = 0.0;
        sliceData();
        for(unsigned int i=0;i<_numTrain;++i){
            std::vector<double> data = (*_trainData)[i];            // extract data point
            double label = data[0];                             // extract point label
            data.erase(data.begin());
            std::vector<double> nLabel = classToOneHot((unsigned int)label);    // encode to 1-of-N

            std::vector<double> outputs = _net->runNet(data);
            error = _net->trainNet(data,nLabel);    // train net, return MSE

            // decode output and compare to correct output
            if( oneHotToClass(outputs) == (unsigned int)label )
                _trueCnt++;
        }
        std::cout << "epoch: " << m << "\t" << "error: " << error << std::endl;

    }

    // stop timer and print out useful info
    int times = timeStop();
    std::cout << "done" << std::endl;
    std::cout << "training time: " << times << " seconds " << std::endl;
    std::cout << "training accuracy: " << _trueCnt*100./(_numTrain * epochs_) << "%" << std::endl;

}

void DataHandler::runNet()
{
    // test net on test data
    timeStart();
    std::cout << "\n" << "test points: " << _numTest << std::endl;
    std::cout << "testing network... ";
    _trueCnt = 0;
    for(unsigned int i=0;i<_numTest;++i)
    {
        std::vector<double> data = (*_testData)[i];     // extract data point
        double label = data[0];                     // extract label
        data.erase(data.begin());
        std::vector<double> outputs = _net->runNet(data);    // run net

        // decode output and compare to correct output
        if( oneHotToClass(outputs) == (unsigned int)label )
            _trueCnt++;
    }

    // stop timer and print out useful info
    int times = timeStop();

    std::cout << "done" << std::endl;
    std::cout << "testing time: " << times << " seconds " << std::endl;
    std::cout << "test accuracy: " << _trueCnt*100./_numTest << "% " << std::endl << std::endl;
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

bool DataHandler::loadParams(char ** params_)
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

    std::cout << "\n" << "total data points: " << _numPts << std::endl;
    std::cout << "learning rate: " << learningRate << std::endl;
    std::cout << "momentum: " << momentum << std::endl;
    std::cout << "weight decay: " << decayRate << std::endl << std::endl;

    return true;
}

bool DataHandler::loadData(DVEC * data_, const char * filename_)
{

    std::cout << "loading data... ";

    std::ifstream inp;
    inp.open(filename_, std::ios::in);
    if(inp){
        for(unsigned int i=0; i<data_->size(); ++i){
            for(unsigned int j=0;j<(*data_)[i].size();++j){
                inp >> (*data_)[i][j];
            }
        }

        inp.close();
    
        std::cout << "done" << std::endl;

        return true;
    }
    else
        return false;
}

bool DataHandler::sliceData()
{

    unsigned int rndIdx = 0;
    std::set<unsigned int> idxSet;

    unsigned int cnt = 0;
    for(unsigned int i=0; i<_numPts; ++i){

        // get a random index in the data that hasn't already been used
        rndIdx = shuffle(_numPts);
        if( idxSet.find(rndIdx) == idxSet.end() )
            idxSet.insert(rndIdx);
        else {
            while( idxSet.find(rndIdx) == idxSet.end() )
                rndIdx = shuffle(_numPts);
            idxSet.insert(rndIdx);
        }

        for(unsigned int j=0;j<_inputs+1;++j){

            if( cnt < _numTrain ){
                if( j==0 )
                    (*_trainData)[cnt][j] = (*_allData)[rndIdx][j];
                else
                    (*_trainData)[cnt][j] = norm((*_allData)[rndIdx][j]);
            }

            if( cnt >= _numTrain && i < (_numTrain + _numTest ) ){
                if( j==0 )
                    (*_testData)[cnt-_numTrain][j] = (*_allData)[rndIdx][j];
                else
                    (*_testData)[cnt-_numTrain][j] = norm((*_allData)[rndIdx][j]);
            }            

            if( cnt > (_numTrain + _numTest ) ){
                if( j==0 )
                    (*_validData)[cnt-_numTrain-_numTest][j] = (*_allData)[rndIdx][j];
                else
                    (*_validData)[cnt-_numTrain-_numTest][j] = norm((*_allData)[rndIdx][j]);
            }

        }

        ++cnt;

    }
    return true;
}

bool DataHandler::saveNet()
{
    bool success = false;
    if(_net){
        _net->saveNet();
        success = true;
    }

    return success;
}

bool DataHandler::loadNet()
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
unsigned int DataHandler::shuffle(const unsigned int& size)
{
  return (unsigned int)(size*(double)rand()/(double)RAND_MAX);
}

double DataHandler::norm(double input)
{
    return input/255.0;
}

void DataHandler::printUsage()
{
    std::cout << std::endl;
    std::cout << "usage is: mnist <learning_rate> <momentum> <decay_rate> <path_to_data>" << std::endl;
    std::cout << std::endl;
}

