CC=g++
CFLAGS=-std=c++0x -g
SOURCES=dfrNeuralTestWrapper_MNISTdigit.cpp dfrNeuralNet.cpp dfrNeuralLayer.cpp dfrNeuralLinearLayer.cpp dfrNeuralSigmoidLayer.cpp dfrNeuralSoftmaxLayer.cpp dfrNeuralTanhLayer.cpp dfrNeuralRectlinLayer.cpp
TARGET=mnist

all: $(TARGET)
    
$(TARGET): $(SOURCES) 
	$(CC) $(CFLAGS) $(SOURCES) -o $(TARGET)

clean:
	rm -f $(TARGET)
