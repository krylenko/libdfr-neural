cmake_minimum_required(VERSION 2.8.3)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -g -Wall")

project(libdfr-neural)

SET(NEURAL_SRC
    src/dfrNeuralLayer.cpp
    src/dfrNeuralLinearLayer.cpp
    src/dfrNeuralRectlinLayer.cpp
    src/dfrNeuralSigmoidLayer.cpp
    src/dfrNeuralSoftmaxLayer.cpp    
    src/dfrNeuralTanhLayer.cpp
    src/dfrNeuralNet.cpp
)  

SET(NEURAL_HEADERS
    src/dfrNeuralLayer.h
    src/dfrNeuralLinearLayer.h
    src/dfrNeuralRectlinLayer.h
    src/dfrNeuralSigmoidLayer.h
    src/dfrNeuralSoftmaxLayer.h    
    src/dfrNeuralTanhLayer.h
    src/dfrNeuralNet.h
)

SET(DATA_SRC
    src/dfrDataHandler.cpp
)

SET(DATA_HEADERS
    src/dfrDataHandler.h
)

SET(MNIST_SRC
    ${NEURAL_SRC}
    ${DATA_SRC}
    src/dfrNeuralTestWrapper_MNISTdigit.cpp
)

add_executable(mnist
    ${MNIST_SRC}
    ${NEURAL_HEADERS}
    ${DATA_HEADERS}
)

#SET(XOR_SRC
#    ${NEURAL_SRC}
#    src/dfrNeuralTestWrapper_XOR.cpp
#)

#add_executable(xor
#    ${XOR_SRC}
#    ${NEURAL_HEADERS}
#)
   
