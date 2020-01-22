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
    const std::string dataFilePath("../../lib/libdfr-neural/train.txt");
    const double dataLimitRatio = 1.0;
    const double testValidateRatio = 1.0;
    const unsigned epochs = 6;
    auto thisSeed = time(nullptr);

    const unsigned long labelLength = 1;
    const unsigned long dataPtLength = INPUT;
    const double dataScaleFactor = 1./255.;

    const bool shuffleData = true;

    const double learningRate = 0.05;
    const double momentum =     0.0;
    const double decayRate =    0.0;
    const double dropoutRate =  0.5;

    std::cout << "initializing network..." << std::endl;
    std::cout << "learning rate: " << "\t \t \t" << learningRate << std::endl;
    std::cout << "momentum: " << "\t \t \t" << momentum << std::endl;
    std::cout << "weight decay: " << "\t \t \t" << decayRate << std::endl;

    // create network and set params, which initializes layers
    NeuralNet DigitNet;
    DigitNet.addLayer(new NeuralTanhLayer(INPUT, HIDDEN_1));
    DigitNet.addLayer(new NeuralSoftmaxLayer(HIDDEN_1, OUTPUT));
    DigitNet.init(learningRate, momentum, decayRate, dropoutRate, thisSeed);

    // load data
    DataLoader* loader(new DataLoader(dataFilePath, labelLength, dataPtLength, dataScaleFactor,
                                      dataLimitRatio, testValidateRatio));
    DataMap_t* holdout = loader->extractHoldoutSet();

    // train
    std::cout << std::endl;
    std::cout << "training network..." << "\t \t" << std::endl;
    std::cout << "epochs: " << epochs << std::endl;
    std::cout << "data points/epoch: " << loader->numTrainPoints() << std::endl;
    std::cout << "total points: " << epochs * loader->numTrainPoints() << std::endl;

    int times = int(time(nullptr));
    auto evalData = DigitNet.train(loader, epochs, shuffleData);
    int timed = int(time(nullptr));
    times = timed - times;

    std::cout << "training time: " << "\t \t \t" << times << " seconds " << std::endl;
    std::cout << "training accuracy: " << "\t \t" << evalData.first << "%" << std::endl;
    std::cout << "training error: " << "\t \t" << evalData.second << std::endl;

    // test
    std::cout << std::endl;
    std::cout << "testing network..." << "\t \t" << std::endl;
    std::cout << "test points: " << "\t \t \t" << holdout->size() << std::endl;

    times = int(time(nullptr));
    double testAccuracy = DigitNet.test(holdout);
    timed = int(time(nullptr));
    times = timed - times;

    std::cout << "testing time: " << "\t \t \t" << times << " seconds " << std::endl;
    std::cout << "test accuracy: " << "\t \t \t" << testAccuracy << "% " << std::endl;
    std::cout << "seed used: " << thisSeed << std::endl;
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
