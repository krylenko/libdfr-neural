// daniel ford 2015
// neural net example script: learning XOR

#include <iostream>
#include <math.h>
#include <stdlib.h>
#include "dfrNeuralNet.h"

std::vector<double> XOR_training();
std::vector<double> softmax_XOR_training();

int main(int argc, char *argv[])
{
  
  // create network
  NeuralNet network;
  
  network.addLayer( new NeuralTanhLayer(2,16) );

  network.addLayer( new NeuralLinearLayer(16,1) );

  network.setParams(0.2, 0, 0);

  const unsigned int iters = 1000;
  const unsigned int testers = 2500;
  int rightCount = 0;

  for(int i=0;i<iters+testers;++i)
  {
  
    double error = 0.0;
    std::vector<double> exor, training;

    // generate training data
    switch(outType)
    {
      case SCALAR:
        exor = XOR_training();
        training.push_back(exor[2]);
        exor.pop_back();
        break;
      case PROB:
        exor = softmax_XOR_training();
        training.push_back(exor[2]);
        training.push_back(exor[3]);
        exor.pop_back();
        exor.pop_back();
        break;
    }

    // training
    if( i<iters )
    {
      std::vector<double> outputs = network.runNet(exor);
      error = network.trainNet(exor,training,outType);
      switch(outType)
      {
        case SCALAR:
          std::cout << exor[0] << "\t" << exor[1] << "\t" << ((int)exor[0]^(int)exor[1]) << "\t" << outputs[0] << "\t" << error << std::endl;
          break;
        case PROB:
          std::cout << exor[0] << "\t" << exor[1] << "\t" << ((int)exor[0]^(int)exor[1]) << "\t" << outputs[0] << "\t" << outputs[1] << "\t" << error << std::endl;
          break;
      }
    }

    // testing
    if( i>=iters )
    {
      std::vector<double> outputs = network.runNet(exor);
      unsigned int out = 0;
      switch(outType)
      {
        case SCALAR:
          out = ( (outputs[0]>0.5) ? 1 : 0 );
          if( out == (int)training[0] )
          {
			      ++rightCount;
		      }
        break;
        case PROB:
		      int classLabel = 0; 
          if( outputs[0] < outputs[1] )
          {
            classLabel = 1;
          }
		      if( 1 == (int)training[classLabel]  )
          {
			      ++rightCount;
		      }
        break;
      }
    }
  }  
  
  std::cout << std::endl << "accuracy: " << 100.0 * rightCount/testers << "%" << std::endl;

  return 0;

}

//auto random XOR generator function
std::vector<double> XOR_training()
{

  double x=0.0;
  std::vector<double> exor;

  x = 1.0 * rand() / (double)RAND_MAX;

  if( x<=0.25 )
  {
    exor.push_back(1.0); exor.push_back(0);
    exor.push_back(1.0);
  }

  if( (x>0.25) && (x<=0.5) )
  {
    exor.push_back(0); exor.push_back(0);
    exor.push_back(0);
  }

  if( (x>0.5) && (x<=0.75) )
  {
    exor.push_back(1.0); exor.push_back(1.0);
    exor.push_back(0);
  }

  if( (x>0.75) && (x<=1.0) )
  {
    exor.push_back(0); exor.push_back(1.0);
    exor.push_back(1.0);
  }

  return exor;
  
}

//auto random XOR generator function (softmax version)
std::vector<double> softmax_XOR_training()
{

  double x=0.0;
  std::vector<double> exor;

  x = 1.0 * rand() / (double)RAND_MAX;

  if( x<=0.25 )
  {
    // inputs
    exor.push_back(1); exor.push_back(0);
    // per-class outputs
    exor.push_back(0); exor.push_back(1);
  }

  if( (x>0.25) && (x<=0.5) )
  {
    exor.push_back(0); exor.push_back(0);
    // per-class outputs
    exor.push_back(1); exor.push_back(0);
  }

  if( (x>0.5) && (x<=0.75) )
  {
    exor.push_back(1); exor.push_back(1);
    // per-class outputs
    exor.push_back(1); exor.push_back(0);
  }

  if( (x>0.75) && (x<=1.0) )
  {
    exor.push_back(0); exor.push_back(1);
    // per-class outputs
    exor.push_back(0); exor.push_back(1);
  }

  return exor;
  
}
