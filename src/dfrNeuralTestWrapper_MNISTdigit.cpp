// daniel ford 2015
// neural net example script: MNIST digit recognition

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <time.h>
#include <vector>
#include <fstream>

#include "dfrNeuralNet.h"
#include "dfrDataHandler.h"

#define FRAMESIZE       (28*28)
#define INPUT           FRAMESIZE
#define HIDDEN          100
#define OUTPUT          10
#define DATA_SIZE       100         // max in file is 42001
#define TRAIN_SIZE      75
#define TEST_SIZE       20
#define EPOCHS          15

int main(int argc, char ** argv)
{

    // init variables
    int truecnt = 0;

    // create network
    NeuralNet * DigitNet = new NeuralNet;
    DigitNet->addLayer( new NeuralTanhLayer(INPUT,HIDDEN) );
    DigitNet->addLayer( new NeuralLinearLayer(HIDDEN,OUTPUT) );
    
    DataHandler handle(argc, argv, DigitNet, INPUT, OUTPUT, DATA_SIZE);

    // create data containers
    DVEC trainData( TRAIN_SIZE, std::vector<double>(INPUT+1,0.0) );
    DVEC testData( TEST_SIZE, std::vector<double>(INPUT+1,0.0) );
    DVEC validData( (DATA_SIZE - TRAIN_SIZE - TEST_SIZE), std::vector<double>(INPUT+1,0.0) );

    std::cout << "training network... ";
    handle.timeStart();

    for(unsigned int m=0; m<EPOCHS; ++m){

        handle.sliceData(trainData, testData, validData, TRAIN_SIZE, TEST_SIZE);
        for(int i=0;i<TRAIN_SIZE;++i){
            std::vector<double> data = trainData[i];            // extract data point
            double label = data[0];                             // extract point label
            data.erase(data.begin());
            std::vector<double> nLabel = handle.classToOneHot((unsigned int)label);    // encode to 1-of-N

            std::vector<double> outputs = DigitNet->runNet(data);
            double error = DigitNet->trainNet(data,nLabel);    // train net, return MSE

            // decode output and compare to correct output
            if( handle.oneHotToClass(outputs) == (unsigned int)label )
                truecnt++;
        }

    }

    // stop timer and print out useful info
    int times = handle.timeStop();
    std::cout << "done" << std::endl;
    std::cout << "training time: " << times << " seconds " << std::endl;
    std::cout << "training accuracy: " << truecnt*100./(TRAIN_SIZE * EPOCHS) << "%" << std::endl;

    // test net on test data
    handle.timeStart();
    std::cout << "\n" << "test points: " << TEST_SIZE << std::endl;
    std::cout << "testing network... ";
    truecnt = 0;
    for(int i=0;i<TEST_SIZE;++i)
    {
        std::vector<double> data = testData[i];     // extract data point
        double label = data[0];                     // extract label
        data.erase(data.begin());
        std::vector<double> outputs = DigitNet->runNet(data);    // run net

        // decode output and compare to correct output
        if( handle.oneHotToClass(outputs) == (unsigned int)label )
            truecnt++;
    }

    // stop timer and print out useful info
    times = handle.timeStop();

    std::cout << "done" << std::endl;
    std::cout << "testing time: " << times << " seconds " << std::endl;
    std::cout << "test accuracy: " << truecnt*100./TEST_SIZE << "% " << std::endl << std::endl;

    // save weights to reuse net in the future
    DigitNet->saveNet();
  
}

