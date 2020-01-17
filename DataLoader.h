#ifndef DATALOADER_H
#define DATALOADER_H

#include <string>
#include <map>
#include <vector>

using DataPt_t = std::pair<std::vector<double>, std::vector<double>>;
using DataVec_t = std::vector<DataPt_t>;
using DataMap_t = std::map<unsigned long, DataPt_t>;

class DataLoader
{
public:
    DataLoader(const std::string filename, const unsigned long dataPtLen,
               const unsigned long labelLen, const double dataScalingFactor,
               const double limitFrac=1.0);
    ~DataLoader();

    unsigned long numPoints() { return numPts; }
    unsigned long numTestPoints() { return numTestPts; }
    unsigned long numTrainPoints() { return numTrainPts; }

    void splitTrainTest(const bool shuffle=false, const double trainTestRatio=0.8);

    DataPt_t trainDataPoint();
    DataPt_t testDataPoint();

private:

    void loadFromFile(const double dataScalingFactor);

    const std::string filename;
    const double limitFrac;
    unsigned long dataLabelLen;
    unsigned long dataPtLen;
    unsigned long numPts;
    unsigned long numTrainPts;
    unsigned long numTestPts;
    DataMap_t::iterator trainIter;
    DataMap_t::iterator testIter;
    DataMap_t allData;
    DataMap_t trainData;
    DataMap_t testData;
};

#endif
