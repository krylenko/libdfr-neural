CC=g++
CFLAGS=-std=c++0x -g
MNIST_SOURCES=dfrNeuralTestWrapper_MNISTdigit.cpp dfrNeuralNet.cpp dfrNeuralLayer.cpp dfrNeuralLinearLayer.cpp dfrNeuralSigmoidLayer.cpp dfrNeuralSoftmaxLayer.cpp dfrNeuralTanhLayer.cpp dfrNeuralRectlinLayer.cpp
XOR_SOURCES=dfrNeuralTestWrapper_XOR.cpp dfrNeuralNet.cpp dfrNeuralLayer.cpp dfrNeuralLinearLayer.cpp dfrNeuralSigmoidLayer.cpp dfrNeuralSoftmaxLayer.cpp dfrNeuralTanhLayer.cpp dfrNeuralRectlinLayer.cpp
TARGET=mnist 

all: $(TARGET)
    
$(TARGET): $(MNIST_SOURCES) 
	$(CC) $(CFLAGS) $(MNIST_SOURCES) -o $(TARGET)

xor: $(XOR_SOURCES)
	$(CC) $(CFLAGS) $(XOR_SOURCES) -o xor

clean:
	rm -f $(TARGET) xor
