#ifndef __DFRDATAHANDLER_H__
#define __DFRDATAHANDLER_H__

// data handler class
// performs:
//   -- parsing of input arguments: training data filename, net params, cross-validation params, type of task (MNIST, XOR, etc.)
//   -- loading of training data
//   -- timing training and inference
//   -- displaying net params, status, and timing to the user
//   -- data slicing, batching, and cross-validation
//   -- feeding data into the net during training and inference
//   -- net loading and saving
//   -- possibly network setup based on task -- define a task w/ its params, don't specify manually

// inputs: argc/argv (training data filename, net params, cross-validation params, type of task), net instance

// a data handler is separate from a network
// the data handler accepts the following inputs:
// -- command line params: net params, file path(s), task (training/inference)
// -- a pointer to a network or a file path to a saved net
// -- a pointer or file path to data (mixed train/test)

#include<vector>

#include "dfrNeuralNet.h"

typedef std::vector< std::vector<double> > DVEC;

class DataHandler
{

    public:

        DataHandler(int argc, char ** params_, NeuralNet * net_, 
                    unsigned int inputs_, unsigned int outputs_, unsigned int numPts_);

        void trainNet(unsigned int epochs_);
        void runNet();

        bool saveNet();

        void printVec(const std::vector<double>& vec);

        ~DataHandler();

    private:

        NeuralNet * _net;
        DVEC * _allData;
        DVEC * _trainData;
        DVEC * _testData;
        DVEC * _validData;
        unsigned int _inputs;
        unsigned int _numClasses;
        unsigned int _numPts;
        unsigned int _numTrain;
        unsigned int _numTest;
        unsigned int _numValid;
        int _timeStart;
        int _timeStop;
        unsigned int _trueCnt;

        bool loadParams(char ** params_);
        bool loadData(DVEC * data_, const char * filename_);
        bool sliceData();

        unsigned int oneHotToClass(const std::vector<double>& oneHot_);
        std::vector<double> classToOneHot(unsigned int class_);

        bool loadNet();

        void timeStart();
        int timeStop();

        unsigned int shuffle(const unsigned int& size);
        double norm(double input);

        void printUsage();

};

#endif
