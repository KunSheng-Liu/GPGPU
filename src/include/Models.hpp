/**
 * \name    Models.hpp
 * 
 * \brief   Declare the model API 
 *          
 * \note    Available model type:
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

/* ************************************************************************************************
 * Macro
 * ************************************************************************************************
 */
#define BENCHMARK( obj, model ) obj.model()

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
    
/* ************************************************************************************************
 * Benchmark
 * ************************************************************************************************
 */
public:
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

    /* Name of model */
    char* modelName;

    /* Number of layer */
    int numOfLayer;


protected:

    /* Number of layer be created */
    static int ModelCount;

    LayerGroup* layerGroup;

    /* Pointer of first/last layer */
    Layer* inputLayer;
    Layer* outputLayer;

};

#endif