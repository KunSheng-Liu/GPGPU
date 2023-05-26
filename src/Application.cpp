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
 * \param   model_type      the model type
 * \param   input_size      [batch = 1, channel, height, width]
 * \param   batch_size      the number of launched tasks when arrival
 * \param   arrival_time    the arrival time: unit (cycle)
 * \param   period          the period : unit (cycle)
 * 
 * \endcond
 * ================================================================================================
 */
Application::Application(char* model_type, vector<int> input_size, int batch_size, unsigned long long arrival_time
                                    , unsigned long long  period, unsigned long long deadline, unsigned long long end_time)
    : appID(appCount++), modelType(model_type), inputSize(input_size), batchSize(batch_size), arrivalTime(arrival_time), period(period)
    , deadline(deadline), endTime(end_time), modelInfo(Model::getModelInfo(model_type)), SM_budget({}), finish(false)
{
    ASSERT(input_size[BATCH] == 1);
    string name = model_type;
    program_name += "_" + to_string(batchSize) + name;
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
    log_T("Application Cycle", modelInfo.modelName);

    if (total_gpu_cycle >= arrivalTime && total_gpu_cycle < endTime)
    {
        int size = inputSize[CHANNEL] * inputSize[HEIGHT] * inputSize[WIDTH];
        for (int i = 0; i < batchSize; i++) tasks.push(Task(total_gpu_cycle, arrivalTime + deadline, appID, vector<unsigned char>(size, 1)));
        arrivalTime += period;
    }

    /* check application finish */
    if(tasks.empty() && runningModels.empty()) finish = true;
}
