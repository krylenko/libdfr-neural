// neural net example script: MNIST digit recognition

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <fstream>

#include "DataLoader.h"
#include "dfrNeuralNet.h"

#define FRAMESIZE       (28*28)
#define INPUT           FRAMESIZE
#define HIDDEN_1        200
#define HIDDEN_2        200
#define OUTPUT          10
#define FROZEN_SEED     1029

// fn prototypes
void printDigit(const vecIntType label, std::vector<double>& data);

int main()
{
    const std::string dataFilePath = "../../lib/libdfr-neural/train.txt";
    const double dataLimitRatio = 0.025;

    const unsigned long labelLength = 1;
    const unsigned long dataPtLength = INPUT;
    const double dataScaleFactor = 1./255.;

    const bool shuffleData = true;

    const double learningRate = 0.05;
    const double momentum =     0.0;
    const double decayRate =    0.0;
    const double dropoutRate =  0.5;

    // init variables
    double error = 0.;
    int truecnt = 0;
    int times, timed;

    std::cout << "initializing network..." << "\t \t" << std::endl;
    std::cout << "learning rate: " << "\t \t \t" << learningRate << std::endl;
    std::cout << "momentum: " << "\t \t \t" << momentum << std::endl;
    std::cout << "weight decay: " << "\t \t \t" << decayRate << std::endl;

    // create network and set params, which initializes layers
    NeuralNet DigitNet;
    DigitNet.addLayer(new NeuralTanhLayer(INPUT, HIDDEN_1));
    DigitNet.addLayer(new NeuralSoftmaxLayer(HIDDEN_1, OUTPUT));
    DigitNet.init(learningRate, momentum, decayRate, dropoutRate);

    // load training and test data
    std::cout << "loading data..." << "\t \t \t";

    DataLoader loader(dataFilePath, labelLength, dataPtLength, dataScaleFactor, dataLimitRatio);
    loader.splitTrainTest(shuffleData);

    std::cout << "train " << loader.numTrainPoints() << ", test " << loader.numTestPoints() <<
                 ", total " << loader.numPoints() << std::endl;

    // loop over training data points and train net
    std::cout << "training network..." << "\t \t" << std::endl;
    times = int(time(nullptr));   // init time counter
    for (vecIntType i = 0; i < loader.numTrainPoints(); ++i) {
        auto thisPt = loader.trainDataPoint();
        auto label = thisPt.first;
        auto data = thisPt.second;
        std::vector<double> nLabel = DigitNet.encodeOneHot(vecIntType(label[0]));    // encode to 1-of-N
        std::vector<double> outputs = DigitNet.runNet(data);
        error = DigitNet.trainNet(data, nLabel);    // train net, return MSE

        // decode output and compare to correct output
        if (DigitNet.decodeOneHot(outputs) == vecIntType(label[0])) {
            truecnt++;
        }
    }

    // stop timer and print out useful info
    timed = int(time(nullptr));
    times = timed - times;
    std::cout << "training time: " << "\t \t \t" << times << " seconds " << std::endl;
    std::cout << "training accuracy: " << "\t \t" << truecnt * 100. / loader.numTrainPoints() << "%" << std::endl;

    // test net on test data
    times = int(time(nullptr));
    std::cout << "\n" << "test points: " << "\t \t \t" << loader.numTestPoints() << std::endl;
    std::cout << "testing network..." << "\t \t" << std::endl;

    truecnt = 0;
    for (vecIntType i = 0; i < loader.numTestPoints(); ++i) {
        auto testPt = loader.testDataPoint();
        auto label = testPt.first;
        auto data = testPt.second;

        std::vector<double> outputs = DigitNet.runNet(data);    // run net

        // decode output and compare to correct output
        if (DigitNet.decodeOneHot(outputs) == vecIntType(label[0])) {
            truecnt++;
        }
    }

    // stop timer and print out useful info
    timed = int(time(nullptr));
    times = timed - times;
    std::cout << "testing time: " << "\t \t \t" << times << " seconds " << std::endl;
    std::cout << "test accuracy: " << "\t \t \t" << truecnt * 100. / loader.numTestPoints() << "% " << std::endl;
}

void printDigit(const vecIntType label, std::vector<double>& data)
{
    std::cout << label << std::endl;
    for (vecIntType j=0; j<FRAMESIZE; ++j) {
        std::cout << (data[j] > 0 ? 1:0);
        if (j % 28 == 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}
