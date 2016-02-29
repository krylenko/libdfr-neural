// daniel ford 2016
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
#define DATA_SIZE       300        // max in file is 42000
#define EPOCHS          15

int main(int argc, char ** argv)
{

    // create network
    NeuralNet * DigitNet = new NeuralNet;
    DigitNet->addLayer( new NeuralTanhLayer(INPUT,HIDDEN) );
    DigitNet->addLayer( new NeuralLinearLayer(HIDDEN,OUTPUT) );
    
    // create handler
    DataHandler handle(argc, argv, DigitNet, INPUT, OUTPUT, DATA_SIZE);
  
    // train and run handler
    handle.trainNet(EPOCHS);
    handle.runNet();

    // save weights to reuse net in the future
    handle.saveNet();
  
}

