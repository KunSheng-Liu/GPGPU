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

    Model(int app_id, const char* model_type, vector<int> input_size, int batch_size = 0);

    ~Model();

/* ************************************************************************************************
 * Type Define
 * ************************************************************************************************
 */
public:
    struct ModelInfo {
        const char* modelName;
        int numOfLayers;
        int ioMemCount;         // unit (Byte)
        int filterMemCount;     // unit (Byte)
        vector<int> inputSize;  // [Channel, Height, Width]
        vector<int> outputSize; // [Channel, Height, Width]

        ModelInfo(const char* model_name) : modelName(model_name) {}
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
    int   getBatchSize  (void) {return batchSize;}
    const char* getModelName  (void) {return modelType;}

    list<Kernel*> findReadyKernels ();
    vector<Kernel>& compileToKernel ();

    pair<int, vector<unsigned char>*> getIFMap  (void) {return modelGraph->getIFMap();}
    pair<int, vector<unsigned char>*> getOFMap  (void) {return modelGraph->getOFMap();}
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

    list<int> SM_budget;
    
    RuntimeRecord recorder;

protected:

    /* Number of layer be created */
    static int modelCount;
    /* Number of layer */
    int numOfLayer;

    /* Batch size of model */
    int batchSize;
    vector<int> inputSize;
    vector<unsigned char> inputData;

    LayerGroup* modelGraph;
    
    vector<Kernel> kernelContainer;
};

#endif