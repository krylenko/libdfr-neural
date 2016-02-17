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

enum{TRAINING,INFERENCE};

typedef std::vector< std::vector<double> > DVEC;

class DataHandler
{

    public:

        DataHandler(int argc, char ** params_, NeuralNet * net_, 
                    unsigned int inputs_, unsigned int outputs_, unsigned int numPts_, unsigned int task_=TRAINING);

        unsigned int oneHotToClass(const std::vector<double>& oneHot_);
        std::vector<double> classToOneHot(unsigned int class_);

        bool sliceData(DVEC& trainData_, DVEC& testData_, DVEC& validData_, unsigned int numTrain_, unsigned int numTest_);

        void timeStart();
        int timeStop();

        void printVec(const std::vector<double>& vec);

        ~DataHandler();

    private:

        NeuralNet * _net;
        DVEC * _allData;
        unsigned int _task;
        unsigned int _inputs;
        unsigned int _numClasses;
        unsigned int _numPts;
        int _timeStart;
        int _timeStop;

        bool _loadParams(char ** params_);
        bool _loadData(DVEC& data_, const char * filename_);

        bool _saveNet();
        bool _loadNet();

        unsigned int _shuffle(const unsigned int& size);
        double _norm(double input);

        void _printUsage();

};

#endif
