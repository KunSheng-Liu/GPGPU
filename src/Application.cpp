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
 * \param   arrival_time    the arrival time                               | unit (cycle)
 * \param   period          the period of tasks                            | unit (cycle)
 * \param   deadline        the deadline of each task from it arrival time | unit (cycle)
 * \param   end_time        the end_time of the application                | unit (cycle)
 * 
 * \note    use char* to store the model_type is because it much easier to print out
 * 
 * \endcond
 * ================================================================================================
 */
Application::Application(char* model_type, vector<int> input_size, int batch_size, unsigned long long arrival_time
                                    , unsigned long long  period /* , unsigned long long deadline */, unsigned long long end_time)
    : appID(appCount++), modelType(model_type), inputSize(input_size), batchSize(batch_size), arrivalTime(arrival_time), period(period)
    /* , deadline(deadline) */, endTime(end_time), SM_budget({}), modelInfo(Model::getModelInfo(model_type)), finish(false)
{
    ASSERT(input_size[BATCH] == 1, "Dimension error");
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
#if (LOG_LEVEL >= TRACE)
    log_T("Application Cycle", modelInfo.modelName);
#endif
    if (arrivalTime < endTime)
    {
        if (total_gpu_cycle >= arrivalTime)
        {
            for (int i = 0; i < batchSize; i++) waitingModels.emplace_back(new Model(appID, modelType, Task(total_gpu_cycle, arrivalTime + deadline, inputSize, vector<DATA_TYPE>(inputSize[CHANNEL] * inputSize[HEIGHT] * inputSize[WIDTH], 1))));
            arrivalTime += period;
        }
    }
    
    /* check application finish */
    else if(waitingModels.empty() && runningModels.empty()) {
        finish = true;
    }
}
