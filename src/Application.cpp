/**
 * \name    Application.cpp
 * 
 * \brief   Implement the API
 * 
 * \date    APR 16, 2023
 */

#include "include/Application.hpp"

/* ************************************************************************************************
 * Global Variable
 * ************************************************************************************************
 */
int Application::appCount = 0;


/** ===============================================================================================
 * \name    Application
 * 
 * \brief   Construct a model
 * 
 * \param   model_type    the model type
 * 
 * \endcond
 * ================================================================================================
 */
Application::Application(char* model_type)
    : appID(appCount++), modelType(model_type), modelInfo(Model::getModelInfo(model_type))
{
    /* Baseline, all application needs to execute once */
    tasks.push(Task(0, 0, appID, vector<unsigned char> (3*224*224, 1)));
}


/** ===============================================================================================
 * \name   ~Application
 * 
 * \brief   Destruct a model
 * 
 * \param   name    the model name
 * 
 * \endcond
 * ================================================================================================
 */
Application::~Application()
{
    // delete mModel;
}


/** ===============================================================================================
 * \name    cycle
 * 
 * \brief   Handling the CPU in a period of cycle
 * 
 * \endcond
 * ================================================================================================
 */
void
Application::cycle()
{
    /* Launch task into queue */
    log_D("Application::cycle", modelInfo.modelName);
}
