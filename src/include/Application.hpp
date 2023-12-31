/**
 * \name    Application.hpp
 * 
 * \brief   Declare the structure of Application
 * 
 * \note    Contains the model and it's data
 * 
 * \date    APR 16, 2023
 */

#ifndef _APPLICATION_HPP_
#define _APPLICATION_HPP_

/* ************************************************************************************************
 * Include Library
 * ************************************************************************************************
 */
#include "App_config.h"
#include "Log.h"

#include "Models.hpp"

/** ===============================================================================================
 * \name    Application
 * 
 * \brief   Contains the model and it's data
 * 
 * \endcond
 * ================================================================================================
 */
class Application
{
/* ************************************************************************************************
 * Class Constructor
 * ************************************************************************************************
 */ 
public:

    Application(char* model_type, vector<int> input_size, int batch_size = 1, unsigned long long arrival_time = 0, unsigned long long  period = -1
        /* , unsigned long long deadline = -1 */, unsigned long long end_time = GPU_F);

   ~Application();

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    void cycle ();

    void setDeadline (unsigned long long _deadline) {deadline = _deadline;}
    
/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
public:
    const int appID;

    const char* modelType;
    
    int batchSize;

    vector<int> inputSize;

    unsigned long long arrivalTime, period, deadline;

    unsigned long long endTime;

    bool finish;

    unordered_set<int> SM_budget;

    /* Model information */
    Model::ModelInfo modelInfo;

    /* Running Models */
    list<Model*> waitingModels = {};
    list<Model*> runningModels = {};

private:
    /* Number of layer be created */
    static int appCount;
};

#endif