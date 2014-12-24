/************************* general-purpose neural network *********************/
/**************** three layers: one input, one hidden, one output *************/
/**************** hidden layer uses tanh activation function ******************/
/*************************** output layer is linear ***************************/

/****************************** weight matrix structure **********************/
//          i,j description
//          n1_curr = neuron 1 of the current layer
//
// 0,0 Wbias_prev->n1_curr         0,1 Wbias_prev->n2_curr     0,n Wbias_prev->nn_curr
// 1,0 Wn1_prev->n1_curr           1,1 Wn1_prev->n2_curr       1,n Wn1_prev->nn_curr
// 2,0 Wn2_prev->n1_curr           2,1 Wn2_prev->n2_curr       2,n Wn2_prev->nn_curr
// n,0 Wnn_prev->n1_curr           n,1 Wnn_prev->n2_curr       n,n Wnn_prev->nn_curr
//
/*****************************************************************************/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#include "libdfr-neural.h"
#include "..\libdfr-matrix\libdfr-matrix.h"

// constructor
NeuralNet::NeuralNet(const int& numIn, const int& numHidden, const int& numOut)

	: 	numInput_(numIn), numHidden_(numHidden), numOutput_(numOut),
		rawError_(0.0), MSE_(0.0),
		learningRate(0.1), momentum(0.1), decayRate(0.0),
		input(1, numInput_+1),
		trainingOut(1,numOutput_),
		computedOut(1,numOutput_),
		weightsHidden(numInput_+1,numHidden_),
		weightsOut(numHidden_+1,numOutput_),
		weightUpdateHidden_(numInput_+1,numHidden),
		deltaHidden_(1,numHidden_+1),
		hiddenOut_(1,numHidden_+1),
		weightUpdateOut_(numHidden_+1,numOutput_),
		deltaOut_(1,numOutput_)
{

	//seed random number generator
	srand(time(NULL));
	
	// randomize weights in the range [-r, r]
	// where r = 1/sqrt(numInput_)
	randomizeWeights_();
	
	// init bias neurons
	input[0][0] = 1.0; hiddenOut_[0][0] = 1.0;

	return;
}

// train neural net (online, one data point at a time)
double NeuralNet::train(const Matrix& data, const Matrix& labeledOut)
{

  // check dimensions
	if( data.cols() != numInput_ ||
      labeledOut.cols() != numOutput_ )
		exit;

	// copy data to input neurons
	for(int i=1;i<numInput_+1;i++)
		input[0][i] = data[0][i-1];

	// copy correct outputs to trainingOut matrix
	trainingOut = labeledOut;
  
  // train the net
  computeOutput_(TRAIN);   // process data input through network
	computeError_();    // (updates MSE_)
	backprop_();        // run backprop to compute weight updates
	updateWeights_();   // apply weight updates

	return MSE_;
  
}

// run neural net (one data point at a time)
void NeuralNet::run(const Matrix& data)
{

  // check dimensions
	if( data.cols() != numInput_ )
		exit;

	// copy data to input neurons
	for(int i=1;i<numInput_+1;i++)
		input[0][i] = data[0][i-1];
  
  // train the net
  computeOutput_(RUN);   // process data input through network
  
}

// load previously trained network weights
// need to add error checking on network size, file pointers
// should also provide ability to load learning rate, network size
int NeuralNet::load(const char fileWeightsHidden[], const char fileWeightsOut[])
{
  weightsHidden.fromFile(fileWeightsHidden);
  weightsOut.fromFile(fileWeightsOut);
}

//randomize hidden, bias, and output neuron weights in the range [-r,r]
//where r = 1/sqrt(number of input nodes)
void NeuralNet::randomizeWeights_()
{

	double r;
	r = 1.0/sqrt(numInput_);

	Matrix wH_temp(weightsHidden);
	wH_temp.random(2*r);
  weightsHidden = wH_temp-r;
	Matrix wO_temp(weightsOut); 
  wO_temp.random(2*r);
	weightsOut = wO_temp-r;
	
}

//propagate input through the network to compute output values
void NeuralNet::computeOutput_(const int& type)
{

	int i,j;
  double dropoutProbHidden = 0.5;
  double dropoutProbInput = 0.2;
  double randNum = 0.;
  double temp = 0.;
  
  // calculate outputs of each neuron in the first layer
  // use dropout on input layer
  for(i=1;i<numHidden_+1;i++)
  {
		hiddenOut_[0][i] = 0;                //init output
		
    //take input, offset by 1 b/c of bias neuron
		for(j=0;j<numInput_+1;j++){      
   
      randNum = (double)rand()/(double)RAND_MAX;
        temp = input[0][j];
      
      if(TRAIN == type)
        hiddenOut_[0][i] += temp * weightsHidden[j][i-1]; //add inputs
      else
        hiddenOut_[0][i] += temp * 0.5 * weightsHidden[j][i-1];
        
    }
        
		hiddenOut_[0][i] = tanh( hiddenOut_[0][i] ); //pass sum to activation function
	}

  //hiddenOut_.print();
  
	// carry forward to output layer (which is linear)
  // use dropout when computing final output
	for(i=0;i<numOutput_;i++)
	{
		computedOut[0][i] = 0;
		
		for(j=0;j<numHidden_+1;j++){
    
      randNum = (double)rand()/(double)RAND_MAX;
      if(randNum > dropoutProbHidden)
          temp = hiddenOut_[0][j];
      else
          temp = 0.;
          
      computedOut[0][i] += temp * weightsOut[j][i]; //sum weights on inputs to this neuron
      
    }

		computedOut[0][i] = tanh( computedOut[0][i] ); //pass sum to activation function

	}
	//computedOut.print();
}

//compute mean squared error between computed output and trained output
void NeuralNet::computeError_()
{

	MSE_ = 0.5 * ( (trainingOut-computedOut)*(trainingOut-computedOut).T() ).sum();

}

//propagate error backward through the network to allocate error to hidden neurons
void NeuralNet::backprop_()
{

	int i,j;

	//compute error of output nodes
	deltaOut_ = trainingOut - computedOut;
  //deltaOut_.print();

	//compute deltas for hidden layer
	deltaHidden_.zeros();
	
	for(i=0; i<numHidden_+1; i++)        //step through all neurons in hidden layer
	{
		for(j=0; j<numOutput_; j++)   //step through all output neuron weights
			deltaHidden_[0][i] += deltaOut_[0][j]*weightsOut[i][j];
	}
}

//update weights of all neurons (with momentum)
void NeuralNet::updateWeights_()
{

	int i,j;
	double W_delta;

    //update weights for hidden layer 1
    for(i=0;i<numInput_+1;i++)
		for(j=0;j<numHidden_;j++)
		{
			//calculate weight change based on delta value and weight decay
			W_delta = 	learningRate * deltaHidden_[0][j] *
						(1.0 - (hiddenOut_[0][j] * hiddenOut_[0][j]) ) *
						input[0][i] -
						learningRate * decayRate * weightsHidden[i][j];
						
			//calculate new weight based on weight change and momentum
      weightsHidden[i][j] += ( W_delta + momentum * weightUpdateHidden_[i][j] );
        
			//save weight change for use in next update
			weightUpdateHidden_[i][j] = W_delta;
		}
	//update weights for output layer
    for(i=0;i<numHidden_+1;i++)
		for(j=0;j<numOutput_;j++)
		{
			W_delta = 	learningRate * deltaOut_[0][j] * hiddenOut_[0][i] -
						learningRate * decayRate * weightsOut[i][j];
						
			weightsOut[i][j] += ( 	W_delta + momentum * 
									weightUpdateOut_[i][j] );
											
			weightUpdateOut_[i][j] = W_delta;
		}
    //weightsHidden.print();
    //weightsOut.print();
}