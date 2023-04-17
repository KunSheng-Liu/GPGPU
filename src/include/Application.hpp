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

/* ************************************************************************************************
 * Type Define
 * ************************************************************************************************
 */
struct task{
    int arrivalTime;
    int deadLine;
    int appIndex;
    vector<unsigned char> data;

    task (int arrival_time, int dead_line, int app_index, vector<unsigned char> data) 
        : arrivalTime(arrival_time), deadLine(dead_line), appIndex(app_index), data(data) {}
};

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

    Application(char* model_type);

   ~Application();

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    void cycle ();
    
/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
public:
    const int appID;

    const char* modelType;

    queue<task> tasks;

private:
    /* Number of layer be created */
    static int appCount;

    /* Model information */
    Model::ModelInfo modelInfo;
};

#endif