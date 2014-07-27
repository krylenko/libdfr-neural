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

#ifndef LIBDFR_NEURAL_H
#define LIBDFR_NEURAL_H

#include "..\libdfr-matrix\libdfr-matrix.h"

class NeuralNet
{

	private:
		
		// network structure variables
		int numInput_;
		int numHidden_;
		int numOutput_;
		
		// error variables
		double rawError_;
		double MSE_;
		
		// internal network arrays
		Matrix weightUpdateHidden_;
		Matrix deltaHidden_;
		Matrix hiddenOut_;
		Matrix weightUpdateOut_;
		Matrix deltaOut_;

		// network computation functions
		void randomizeWeights_();
		void computeOutput_();
		void computeError_(const int& epoch);
		void backprop_();
		void updateWeights_();

	public:

		NeuralNet(	const int& numIn,	// constructor
					const int& numHidden,
					const int& numOut);
		//~NeuralNet();					// destructor; may not need
	
		Matrix input;			// network arrays
		Matrix weightsHidden;
		Matrix weightsOut;
		Matrix trainingOut;
		Matrix computedOut;
		
		double learningRate;   	// controls rate of weight updates
								// range [0,1]

		double momentum;        // controls how much of the previous weight update
								// is added to the current update
								// range [0,1]

		double decayRate;      	//controls how fast the weights decay during update
								//essentially a "forgetting" term
								//range [0,1]

		// network I/O
		void putInput(const Matrix& netInput);
		void putTraining(const Matrix& training);
								
		// main network functions
		Matrix activate();
		double train(const int& epoch);
								
};

#endif
