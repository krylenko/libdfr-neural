#include "DataLoader.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <utility>

DataLoader::DataLoader(const std::string filename, const unsigned long labelLen,
                       const unsigned long dataPtLen, const double dataScalingFactor,
                       const double limitFrac, const double trainValidateFrac)
    : filename(filename)
    , limitFrac((limitFrac < 0.0) ? 1.0 : limitFrac)
    , trainValFrac(trainValidateFrac < 0.0 ? 1.0 : trainValidateFrac)
    , dataLabelLen(labelLen)
    , dataPtLen(dataPtLen)
    , numPts(0)
    , numTrainPts(0)
    , numTestPts(0)
{
    loadFromFile(dataScalingFactor);
    splitTrainTest();
}

DataLoader::~DataLoader() {

}

DataPt_t DataLoader::trainDataPoint()
{
    DataPt_t thisPt;
    if (trainIter != trainData.end()) {
        thisPt = trainIter->second;
        ++trainIter;
    }
    return thisPt;
}

DataPt_t DataLoader::testDataPoint()
{
    DataPt_t thisPt;
    if (testIter != testData.end()) {
        thisPt = testIter->second;
        ++testIter;
    }
    return thisPt;
}

void DataLoader::loadFromFile(const double dataScalingFactor)
{
    unsigned long index = 0;
    std::vector<double> ptLabel(dataLabelLen, 0.0), dataPt(dataPtLen, 0.0);
    std::ifstream inp;
    inp.open(filename, std::ios::in);
    if (inp) {
        while (!inp.eof()) {
            for (unsigned i = 0; i < dataLabelLen; ++i) {
                if (!inp.eof()) {
                    inp >> ptLabel.at(i);
                }
            }
            for (unsigned j = 0; j < dataPtLen; ++j) {
                if (!inp.eof()) {
                    inp >> dataPt.at(j);
                    dataPt[j] = dataPt[j] * dataScalingFactor;
                }
            }
            if (!inp.eof()) {
                allData[index] = std::make_pair(ptLabel, dataPt);
                ++index;
            }
        }
        inp.close();
    }
    // return only the specified fraction of the full data set
    if (limitFrac < 1.0) {
        unsigned long endIndex = static_cast<unsigned long>(limitFrac * allData.size() - 0.5);
        allData.erase(allData.find(++endIndex), allData.end());
    }
    numPts = allData.size();
}

void DataLoader::splitTrainTest(const bool shuffle)
{
    unsigned long trainLen = static_cast<unsigned long>(trainValFrac * numPts);
    unsigned long testLen = numPts - trainLen;

    trainData.clear();
    testData.clear();

    if (shuffle) {
        std::map<unsigned long, unsigned long> usedIndices;
        // place a random selection of the appropriate size in testData
        while (testData.size() < testLen) {
            unsigned long randIdx =
                    static_cast<unsigned long>(allData.size() * double(rand()) / double(RAND_MAX));
            if (usedIndices.find(randIdx) == usedIndices.end()) {
                testData[randIdx] = allData.at(randIdx);
                usedIndices[randIdx] = randIdx;
            }
        }
        // shove the rest into trainData
        for (auto it = allData.begin(); it != allData.end(); ++it) {
            auto thisPt = *it;
            auto thisPtIdx = thisPt.first;
            if (usedIndices.find(thisPtIdx) == usedIndices.end()) {
                trainData[thisPtIdx] = thisPt.second;
                usedIndices[thisPtIdx] = thisPtIdx;
            }
        }
    } else {
        for (auto it = allData.begin(); it != allData.end(); ++it) {
            auto thisPt = *it;
            if (trainData.size() < trainLen) {
                trainData[thisPt.first] = thisPt.second;
            } else {
                testData[thisPt.first] = thisPt.second;
            }
        }
    }

    trainIter = trainData.begin();
    testIter = testData.begin();
    numTrainPts = trainData.size();
    numTestPts = testData.size();

    assert(numTrainPts == trainLen);
    assert(numTestPts == testLen);
}

std::shared_ptr<DataMap_t> DataLoader::extractHoldoutSet(const double sizeFrac, const bool shuffle)
{
    std::shared_ptr<DataMap_t> holdout(new DataMap_t);
    const unsigned long holdoutPts = static_cast<unsigned long>(sizeFrac * numPts);
    const unsigned long fullSizeIdx = allData.size() - 1;

    unsigned long ptIdx = 0;
    for (unsigned long p = 0; p < holdoutPts; ++p) {
        if (shuffle) {
            ptIdx = static_cast<unsigned long>(fullSizeIdx * double(rand()) / RAND_MAX);
            while (holdout->find(ptIdx) != holdout->end()) {
                ptIdx = static_cast<unsigned long>(fullSizeIdx * double(rand()) / RAND_MAX);
            }
        } else {
            ptIdx = fullSizeIdx - p;
        }
        (*holdout)[ptIdx] = allData.at(ptIdx);
        allData.erase(allData.find(ptIdx));
    }
    numPts = allData.size();
    splitTrainTest();
    return holdout;
}
