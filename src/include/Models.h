/**
 * \name    Models.h
 * 
 * \brief   Declare the model API 
 *          
 * \note    Available model type:
 *          - \b RESNET18
 * 
 * \date    Mar 31, 2023
 */

#ifndef _MODLES_H_
#define _MODLES_H_

/* ************************************************************************************************
 * Include Library
 * ************************************************************************************************
 */
#include "App_config.h"
#include "Layer.h"
#include "Log.h"

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

    void printSummary (void);


/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
protected:
    /* The index of model. Each model have a unique index */
    int modelIndex;

    /* Number of layer */
    int numOfLayer;

    /* Pointer of first/last layer */
    Layer* inputLayer;
    Layer* outputLayer;

};

#endif