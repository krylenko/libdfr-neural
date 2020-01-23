#ifndef DATALOADER_H
#define DATALOADER_H

#include <string>
#include <map>
#include <memory>
#include <vector>

using DataPt_t = std::pair<std::vector<double>, std::vector<double>>;
using DataVec_t = std::vector<DataPt_t>;
using DataMap_t = std::map<unsigned long, DataPt_t>;

class DataLoader
{
public:
    DataLoader(const std::string filename, const unsigned long dataPtLen,
               const unsigned long labelLen, const double dataScalingFactor,
               const double limitFrac=1.0, const double trainValidateFrac=1.0);
    ~DataLoader();

    std::shared_ptr<DataMap_t> extractHoldoutSet(const double sizeFrac=0.1, const bool shuffle=true);
    void splitTrainTest(const bool shuffle=false);

    DataPt_t trainDataPoint();
    DataPt_t testDataPoint();

    unsigned long numPoints() { return numPts; }
    unsigned long numTestPoints() { return numTestPts; }
    unsigned long numTrainPoints() { return numTrainPts; }

private:

    void loadFromFile(const double dataScalingFactor);

    const std::string filename;
    const double limitFrac;
    const double trainValFrac;
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
