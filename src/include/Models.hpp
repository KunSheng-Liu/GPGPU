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

    Model();

    ~Model();


/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:

    void memoryAllocate (MMU* mmu);
    void printSummary ();

    void setBatchSize (int batch_size);

    char* getModelName (void) {return modelName;}
    int   getNumOfLayer (void) {return numOfLayer;}

    vector<int>* getIFMapSize  (void) {return layerGroup->getIFMapSize();}
    vector<int>* getOFMapSize  (void) {return layerGroup->getOFMapSize();}
    
/* ************************************************************************************************
 * Benchmark
 * ************************************************************************************************
 */
public:
    void None();
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
    /* The index of model. Each model have a unique index */
    const int modelIndex;

protected:

    /* Number of layer be created */
    static int modelCount;

    /* Name of model */
    char* modelName;

    /* Number of layer */
    int numOfLayer;

    /* Batch size of model */
    int batchSize = 0;

    /* Memory information of model */
    int ioMemCount = 0;
    int filterMemCount = 0;

    LayerGroup* layerGroup;
};

#endif