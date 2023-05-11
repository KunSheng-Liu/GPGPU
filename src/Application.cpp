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
    , SM_budget(0), finish(false)
{
    /* Baseline, all application needs to execute once */
    tasks.push(Task(total_gpu_cycle, -1, appID, vector<unsigned char> (3*224*224, 1)));
}


/** ===============================================================================================
 * \name   ~Application
 * 
 * \brief   Destruct a pplication
 * 
 * \endcond
 * ================================================================================================
 */
Application::~Application()
{
    ASSERT(runningModels.empty(), "Error Destruct")
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
    log_D("Application Cycle", modelInfo.modelName);

    /* check application finish */
    if(tasks.empty() && runningModels.empty()) finish = true;
}
