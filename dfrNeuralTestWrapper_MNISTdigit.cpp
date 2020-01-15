// daniel ford 2015
// neural net example script: MNIST digit recognition

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <time.h>
#include <vector>
#include <fstream>

#include "dfrNeuralNet.h"

#define FRAMESIZE       (28*28)
#define INPUT           FRAMESIZE
#define HIDDEN_1        200
#define HIDDEN_2        200
#define OUTPUT          10
#define DATA_SIZE       30000         // max in file is 42001
#define TRAIN_SIZE      1500
#define TEST_SIZE       25
#define FROZEN_SEED     13

// fn prototypes
std::vector<double> encode(const vecIntType digit);
vecIntType decode(std::vector<double>& netOut);
void buildData(const std::vector< std::vector<double> >& allData, std::vector< std::vector<double> >& trainData,
               const vecIntType trainSize, std::vector< std::vector<double> >& testData, const vecIntType testSize,
               const bool shuffle);
vecIntType shuffleIdx(const vecIntType size);
void loadFromFile(std::vector< std::vector<double> >& vec, const char filename[]);

int main()
{

    // init variables
    double error = 0.;
    int truecnt = 0;
    int times, timed;

    // print useful info for reference
    //std::cout << "\n" << "hidden neurons: " << "\t \t" << HIDDEN << std::endl;

    // init random number generator
    srand(unsigned(time(nullptr)));

    // create network
    std::cout << "initializing network..." << "\t \t";
    NeuralNet DigitNet;

    NeuralLayer * pHiddenLayer1 = new NeuralTanhLayer(INPUT, HIDDEN_1, FROZEN_SEED);
    DigitNet.addLayer(pHiddenLayer1);
    NeuralLayer * pOutputLayer = new NeuralSoftmaxLayer(HIDDEN_1, OUTPUT, FROZEN_SEED);
    DigitNet.addLayer(pOutputLayer);
    const unsigned int outType = PROB;

    // set learning rate, momentum, decay rate
    const double learningRate = 0.05;
    const double momentum =     0.0;
    const double decayRate =    0.0;
    DigitNet.setParams(learningRate, momentum, decayRate, outType);
    std::cout << "done" << std::endl;

    // load training and test data
    std::cout << "loading data..." << "\t \t \t";

    std::vector< std::vector<double> > bigData(DATA_SIZE, std::vector<double>(INPUT+1,0.0));
    std::vector< std::vector<double> > trainData(TRAIN_SIZE, std::vector<double>(INPUT+1,0.0));
    std::vector< std::vector<double> > testData(TEST_SIZE, std::vector<double>(INPUT+1,0.0));

    const bool shuffleData = false;
    loadFromFile(bigData, "../../lib/libdfr-neural/train.txt");
    buildData(bigData, trainData, TRAIN_SIZE, testData, TEST_SIZE, shuffleData);

    /*
    std::cout << std::endl;
    for (int idx = 0; idx < 5; idx++) {
        auto label = trainData[idx][0];
        std::cout << label << std::endl;
        for (int j = 1; j<FRAMESIZE; ++j) {
            std::cout << (trainData[idx][j] > 0 ? 1:0);
            if (j % 28 == 0) {
                std::cout << std::endl;
            }
        }
        std::cout << std::endl;
    }
    */

    std::cout << "done" << std::endl;

    // loop over training data points and train net
    // slice off first column of each row (example)
    times = int(time(nullptr));   // init time counter
    std::cout << "\n" << "training examples: " << "\t \t" << TRAIN_SIZE << std::endl;
    std::cout << "learning rate: " << "\t \t \t" << learningRate << std::endl;
    std::cout << "momentum: " << "\t \t \t" << momentum << std::endl;
    std::cout << "weight decay: " << "\t \t \t" << decayRate << std::endl;
    std::cout << "training network..." << "\t \t";

    for (vecIntType i=0; i<TRAIN_SIZE; ++i) {
        std::vector<double> data = trainData[i];            // extract data point
        double label = data[0];                             // extract point label
        data.erase(data.begin());
        std::vector<double> nLabel = encode(vecIntType(label));    // encode to 1-of-N

        /*
        std::cout << label << std::endl;
        for (int j=0; j<FRAMESIZE; ++j) {
            std::cout << (data[j] > 0 ? 1:0);
            if (j % 28 == 0) {
                std::cout << std::endl;
            }
        }
        std::cout << std::endl;
        */

        std::vector<double> outputs = DigitNet.runNet(data);
        error = DigitNet.trainNet(data, nLabel, outType);    // train net, return MSE

        // decode output and compare to correct output
        if (decode(outputs) == vecIntType(label)) {
            truecnt++;
        }
    }

    // stop timer and print out useful info
    timed = int(time(nullptr));
    times = timed - times;
    std::cout << "done" << std::endl;
    std::cout << "training time: " << "\t \t \t" << times << " seconds " << std::endl;
    std::cout << "training accuracy: " << "\t \t" << truecnt * 100. / TRAIN_SIZE << "%" << std::endl;

    // test net on test data
    times = int(time(nullptr));   // init time counter
    std::cout << "\n" << "test points: " << "\t \t \t" << TEST_SIZE << std::endl;
    std::cout << "testing network..." << "\t \t";
    truecnt = 0;
    for (vecIntType i=0; i<TEST_SIZE; ++i) {

        std::vector<double> data = testData[i];     // extract data point
        double label = data[0];                     // extract label
        data.erase(data.begin());

        std::vector<double> outputs = DigitNet.runNet(data);    // run net

        /*
        if (i % 10 == 0) {
            std::cout << int(label) << " -> " << decode(outputs) << std::endl;
        }
        */

        // decode output and compare to correct output
        if (decode(outputs) == vecIntType(label)) {
            truecnt++;
        }

    }

    // stop timer and print out useful info
    timed = int(time(nullptr));
    times = timed - times;
    std::cout << "done" << std::endl;
    std::cout << "testing time: " << "\t \t \t" << times << " seconds " << std::endl;
    std::cout << "test accuracy: " << "\t \t \t" << truecnt*100./TEST_SIZE << "% " << std::endl;

    // save weights to reuse net in the future
    DigitNet.saveNet();

}

// separate data into training and test sets
void buildData(const std::vector< std::vector<double> >& allData, std::vector< std::vector<double> >& trainData,
               const vecIntType trainSize, std::vector< std::vector<double> >& testData, const vecIntType testSize,
               const bool shuffle)
{

    vecIntType rndIdx = 0;

    // extract training data
    for (vecIntType i=0; i<trainSize; ++i) {
        rndIdx = shuffle? shuffleIdx(trainSize) : i;
        for (vecIntType j=0; j<INPUT+1; ++j) {
            trainData[i][j] = (j == 0) ? allData[rndIdx][j] : allData[rndIdx][j] / 255.;
        }
    }

    // extract test data
    for (vecIntType i=0; i<testSize; ++i) {
        rndIdx = shuffle? shuffleIdx(testSize) : i;
        for (vecIntType j=0; j<INPUT+1; ++j)
        testData[i][j] = (j == 0) ? allData[rndIdx][j] : allData[rndIdx][j] / 255.;
    }

}

// return random index for data of given size in range [0, size-1]
vecIntType shuffleIdx(const vecIntType size)
{
    return vecIntType(size * double(rand()) / double(RAND_MAX));
}

// convert single input digit label into 1-of-10 encoded matrix
std::vector<double> encode(const vecIntType digit)
{
    std::vector<double> output(10, 0.);
    output[digit] = 1.;
    return output;
}

// convert 1-of-10 float output matrix to single digit
vecIntType decode(std::vector<double>& netOut)
{
    vecIntType digit = 0; double tmp = netOut[0];
    for (vecIntType i = 0;i<10;++i) {
        if (netOut[i] > tmp) {
            digit = i;
            tmp = netOut[i];
        }
    }
    return digit;
}

void loadFromFile(std::vector< std::vector<double> >& vec, const char filename[])
{
    std::ifstream inp;
    inp.open(filename, std::ios::in);
    if (inp) {
        for (unsigned int i=0; i<vec.size(); ++i) {
            for (unsigned int j=0; j<vec[0].size(); ++j) {
                inp >> vec[i][j];
            }
        }
        inp.close();
    }
}