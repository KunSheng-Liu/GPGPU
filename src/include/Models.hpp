/**
 * \name    Models.hpp
 * 
 * \brief   Declare the model API 
 *          
 * \note    Available model type:
 *          - \b VGG16
 *          - \b RESNET18
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

    Model(int app_id);
    Model(int app_id, int batch_size);

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

    void memoryAllocate (MMU* mmu);
    vector<Kernel>& compileToKernel ();
    void printSummary ();

    void setBatchSize (int batch_size);
    list<Kernel*> findReadyKernels();

    char* getModelName (void) {return modelName;}
    int   getNumOfLayer (void) {return numOfLayer;}

    vector<int>* getIFMapSize  (void) {return modelGraph->getIFMapSize();}
    vector<int>* getOFMapSize  (void) {return modelGraph->getOFMapSize();}

    static ModelInfo getModelInfo (const char* model_type);
    
/* ************************************************************************************************
 * Benchmark
 * ************************************************************************************************
 */
public:
    void buildLayerGraph (const char* model_type);

    void Test();
    // void LeNet();
    void VGG16();
    // void VGG19();
    void ResNet18();
    // void GoogLeNet();

/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
public:
    /* The index of source application. */
    const int appID;

    /* The index of model. Each model have a unique index */
    const int modelID;

protected:

    /* Number of layer be created */
    static int modelCount;

    /* Name of model */
    char* modelName;

    /* Number of layer */
    int numOfLayer;

    /* Batch size of model */
    int batchSize = 0;

    LayerGroup* modelGraph;
    
    vector<Kernel> kernelContainer;
};

#endif