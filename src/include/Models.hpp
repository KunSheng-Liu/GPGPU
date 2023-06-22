/**
 * \name    Models.hpp
 * 
 * \brief   Declare the model API 
 *          
 * \note    Available model type:
 *          - \b Test
 *          - \b LeNet
 *          - \b ResNet18
 *          - \b VGG16
 *          = \b GoogleNet
 * 
 * \date    APR 4, 2023
 */

#ifndef _MODLES_HPP_
#define _MODLES_HPP_

/* ************************************************************************************************
 * Include Library
 * ************************************************************************************************
 */
#include "App_config.h"
#include "Log.h"

#include "Kernel.hpp"
#include "Layers.hpp"
#include "LayerGroup.hpp"

/* ************************************************************************************************
 * Type Define
 * ************************************************************************************************
 */
struct Task{
    const int arrivalTime, deadLine;
    vector<int> inputSize;
    vector<DATA_TYPE> data;

    Task (int arrival_time, int dead_line, vector<int> input_size, vector<DATA_TYPE> data) 
        : arrivalTime(arrival_time), deadLine(dead_line), inputSize(input_size), data(data) {}
};

/** ===============================================================================================
 * \name    Model
 * 
 * \brief   The base class of NN model. You can add new layer type by inheritance this class and
 *          override the virtual function to fit the desire.
 * \endcond
 * ================================================================================================
 */
class Model
{
/* ************************************************************************************************
 * Class Constructor
 * ************************************************************************************************
 */ 
public:

    Model(int app_id, const char* model_type, Task task);

   ~Model();

/* ************************************************************************************************
 * Type Define
 * ************************************************************************************************
 */
public:
    struct ModelInfo {
        const char* modelName;
        unsigned numOfLayers;
        unsigned numOfRequest;
        unsigned long long numOfCycle;
        unsigned long long ioMemCount;      // unit (Byte)
        unsigned long long filterMemCount;  // unit (Byte)
        unsigned long long numOfRead;       // unit (Byte)
        unsigned long long numOfWrite;      // unit (Byte)
        vector<int> inputSize;              // [Channel, Height, Width]
        vector<int> outputSize;             // [Channel, Height, Width]

        unsigned long long totalExecuteTime;
        vector<unsigned long long> layerExecuteTime;

        ModelInfo& operator+= (const Kernel::KernelInfo& other) {
        numOfRead    += other.numOfRead;
        numOfWrite   += other.numOfWrite;
        numOfCycle   += other.numOfCycle;
        numOfRequest += other.numOfRequest;
        return *this;
        }

        ModelInfo(const char* model_name = (char*)"Null") : modelName(model_name) 
            , numOfLayers(0), numOfRequest(0), numOfCycle(0), ioMemCount(0), filterMemCount(0)
            , numOfRead(0), numOfWrite(0), inputSize({}), outputSize({}), totalExecuteTime(0), layerExecuteTime({}) {}
    };

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:

    void setBatchSize (int batch_size);
    void memoryAllocate (MMU* mmu);
    PageRecord memoryRelease  (MMU* mmu);

    void printSummary ();

    bool checkFinish ();

    int   getNumOfLayer (void) {return numOfLayer;}
    int   getBatchSize  (void) {return task.inputSize[BATCH];}
    const char* getModelName  (void) {return modelType;}

    list<Kernel*> findReadyKernels ();
    list<Kernel*> getRunningKernels ();
    vector<bool>  getKernelStatus ();
    
    vector<Kernel>& compileToKernel ();

    pair<int, vector<DATA_TYPE>*> getIFMap  (void) {return modelGraph->getIFMap();}
    pair<int, vector<DATA_TYPE>*> getOFMap  (void) {return modelGraph->getOFMap();}
    vector<int> getIFMapSize  (void) const  {return modelGraph->getIFMapSize();}
    vector<int> getOFMapSize  (void) const  {return modelGraph->getOFMapSize();}

    static ModelInfo getModelInfo (const char* model_type);
    
/* ************************************************************************************************
 * Benchmark
 * ************************************************************************************************
 */
public:
    void buildLayerGraph ();

    void LeNet     (vector<int> = {1, 1,  32,  32});
    void CaffeNet  (vector<int> = {1, 3, 227, 227});
    void ResNet18  (vector<int> = {1, 3, 224, 224});
    void VGG16     (vector<int> = {1, 3, 224, 224});
    // void VGG19  ();
    void GoogleNet (vector<int> = {1, 3, 224, 224});

/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
public:
    /* The index of source application. */
    const int appID;

    /* The index of model. Each model have a unique index */
    const int modelID;

    /* Name of model */
    const char* modelType;

	unsigned long long startTime = -1, endTime = -1;

    Task task;

    unordered_set<int> SM_budget;
    
    RuntimeRecord recorder;

    PageRecord page_record;

protected:

    /* Number of layer be created */
    static int modelCount;

    /* Number of layer */
    int numOfLayer;

    LayerGroup* modelGraph;
    
    vector<Kernel> kernelContainer;
};

#endif