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
#define HIDDEN          100
#define OUTPUT          10
#define DATA_SIZE       40000         // max in file is 42001
#define TRAIN_SIZE      20000
#define TEST_SIZE       5000

// fn prototypes
std::vector<double> encode(const int& digit);
int decode(std::vector<double>& netOut);
void buildData(const std::vector< std::vector<double> >& allData, std::vector< std::vector<double> >& trainData, const int& trainSize, std::vector< std::vector<double> >& testData, const int& testSize);
int shuffle(const int& size);
void loadFromFile(std::vector< std::vector<double> >& vec, const char filename[]);

int main()
{

  // init variables
  double error = 0.;
  int truecnt = 0;
  int times,timed;
  
  // init random number generator
  srand((int)time(NULL));  

  // create network
  std::cout << std::endl << "initializing network..." << "\t \t";
  NeuralNet DigitNet;

  DigitNet.addLayer( new NeuralRectlinLayer(INPUT,HIDDEN) );

  /*
  int numHidden = 2;
  numHidden /= 2;
  for(auto i=0;i<numHidden;++i)
  {
    DigitNet.addLayer( new NeuralRectlinLayer(HIDDEN,HIDDEN*2) );
    DigitNet.addLayer( new NeuralRectlinLayer(HIDDEN*2,HIDDEN) );
  }
  */

  NeuralLayer * pOutputLayer = new NeuralSigmoidLayer(HIDDEN,OUTPUT);
  DigitNet.addLayer( pOutputLayer );

  // set output type:
  // SCALAR = tanh or sigmoid output layer (use one output neuron)
  // PROB = softmax output layer, 1-of-N output encoding (use n output neurons, one per class)
  const unsigned int outType = PROB;

  // set learning rate, momentum, decay rate
  const double learningRate = 0.05;
  const double momentum =     0.0;
  const double decayRate =    0.0;
  DigitNet.setParams(learningRate,momentum,decayRate,outType);

  std::cout << "done" << std::endl;

  // print useful info for reference
  std::cout << "hidden neurons: " << "\t \t" << HIDDEN << std::endl;
  std::cout << "hidden layers: " << "\t \t \t" << DigitNet.numLayers()-2 << std::endl << std::endl;
  
  // load training and test data
  std::cout << "loading data..." << "\t \t \t";
  std::vector< std::vector<double> > bigData( DATA_SIZE,std::vector<double>(INPUT+1,0.0) );
  loadFromFile(bigData,"train.txt");

  std::vector< std::vector<double> > trainData( TRAIN_SIZE,std::vector<double>(INPUT+1,0.0) );
  std::vector< std::vector<double> > testData( TEST_SIZE,std::vector<double>(INPUT+1,0.0) );
  
  buildData(bigData,trainData,TRAIN_SIZE,testData,TEST_SIZE);
  std::cout << "done" << std::endl;
  
  // loop over training data points and train net
  // slice off first column of each row (example)
  times=(int)time(NULL);   // init time counter
  std::cout << "\n" << "training examples: " << "\t \t" << TRAIN_SIZE << std::endl;
  std::cout << "learning rate: " << "\t \t \t" << learningRate << std::endl;
  std::cout << "momentum: " << "\t \t \t" << momentum << std::endl;
  std::cout << "weight decay: " << "\t \t \t" << decayRate << std::endl << std::endl;
  std::cout << "training network..." << "\t \t";
  for(int i=0;i<TRAIN_SIZE;++i)
  {
    std::vector<double> data = trainData[i];            // extract data point
    double label = data[0];                             // extract point label
    data.erase(data.begin());
    std::vector<double> nLabel = encode((int)label);    // encode to 1-of-N   
    
    std::vector<double> outputs = DigitNet.runNet(data);
    error = DigitNet.trainNet(data,nLabel,outType);    // train net, return MSE
    std::cout << error << std::endl;

    // decode output and compare to correct output 
    //std::cout << label << " ... " << std::endl;
    if( decode(outputs) == (int)label )
        truecnt++;    
  }

  // stop timer and print out useful info
  timed=(int)time(NULL);
  times=timed-times;
  std::cout << "done" << std::endl;
  std::cout << "training time: " << "\t \t \t" << times << " seconds " << std::endl;
  std::cout << "training accuracy: " << "\t \t" << truecnt*100./TRAIN_SIZE << "%" << std::endl;
  
  // test net on test data
  times=(int)time(NULL);   // init time counter
  std::cout << "\n" << "test points: " << "\t \t \t" << TEST_SIZE << std::endl;
  std::cout << "testing network..." << "\t \t";
  truecnt = 0;
  for(int i=0;i<TEST_SIZE;++i)
  {
    
    std::vector<double> data = testData[i];     // extract data point 
    double label = data[0];                     // extract label
    data.erase(data.begin());
   
    std::vector<double> outputs = DigitNet.runNet(data);    // run net

    // decode output and compare to correct output
    if( decode(outputs) == (int)label )
        truecnt++;    
    
  }

  // stop timer and print out useful info
  timed=(int)time(NULL);
  times=timed-times;
  std::cout << "done" << std::endl;
  std::cout << "testing time: " << "\t \t \t" << times << " seconds " << std::endl;
  std::cout << "test accuracy: " << "\t \t \t" << truecnt*100./TEST_SIZE << "% " << std::endl << std::endl;
  
  // save weights to reuse net in the future
  DigitNet.saveNet();
  
}

// separate data into training and test sets
void buildData( const std::vector< std::vector<double> >& allData,
                std::vector< std::vector<double> >& trainData,
                const int& trainSize, 
                std::vector< std::vector<double> >& testData,
                const int& testSize)
{

  int rndIdx = 0;

  // extract training data
  for(int i=0;i<trainSize;++i)
  {
    rndIdx = shuffle(trainSize);    
    for(int j=0;j<INPUT+1;++j)
      if( j == 0 )
        trainData[i][j] = allData[rndIdx][j];
      else
        trainData[i][j] = allData[rndIdx][j]/255.0;
  }

  // extract test data
  for(int i=0;i<testSize;++i)
  {
    rndIdx = shuffle(testSize);
    for(int j=0;j<INPUT+1;++j)
      if( j == 0 )
        testData[i][j] = allData[rndIdx][j];
      else
        testData[i][j] = allData[rndIdx][j]/255.0;
  }
  
}

// return random index for data of given size in range [0,size-1]
int shuffle(const int& size)
{
  return (int)(size*(double)rand()/(double)RAND_MAX);
}

// convert single input digit label into 1-of-10 encoded matrix
std::vector<double> encode(const int& digit)
{

  std::vector<double> output(10,-1.0);
  output[digit] = 1.;
  return output;
  
}

// convert 1-of-10 float output matrix to single digit
int decode(std::vector<double>& netOut)
{
  int digit = 0; double tmp = netOut[0];
  for(int i = 0;i<OUTPUT;++i)
  {
    //std::cout << netOut[i] << " /// ";
    if(netOut[i] > tmp)
    {
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
  if(inp)
  {
    for(unsigned int i=0; i<vec.size(); ++i)
    {
      for(unsigned int j=0;j<vec[0].size();++j)
      {
        inp >> vec[i][j];
      }
    }

    inp.close();
  }
}
