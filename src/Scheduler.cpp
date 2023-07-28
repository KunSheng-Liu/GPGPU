/**
 * \name    Scheduler.cpp
 * 
 * \brief   Implement the basic function used in CPU.hpp.
 * 
 * \date    Jun 22, 2023
 */
#include "include/Scheduler.hpp"


/** ===============================================================================================
 * \name    Scheduler
 * 
 * \brief   Set up callback function
 * 
 * \endcond
 * ================================================================================================
 */
Scheduler::Scheduler (CPU* cpu) : mCPU(cpu)
{

}

/** ===============================================================================================
 * \name    Kernel_Launcher
 * 
 * \brief   Launch kernel to GPU
 * 
 * \endcond
 * ================================================================================================
 */
void 
Scheduler::kernelLauncher (Kernel* kernel)
{
    ASSERT(!kernel->SM_List->empty(), "Kernel has no computing resource");

    if (kernel->compileRequest(&mCPU->mMMU))
    {
        ASSERT(mCPU->mGPU->launchKernel(kernel), "Failed launch kernel");
        kernel->startCycle = total_gpu_cycle;
        kernel->running    = true;
    } 
    else log_I("compileRequest", "kernel: " + to_string(kernel->kernelID) + "has empty requests");
}


/** ===============================================================================================
 * \name    missDeadlineHandler
 * 
 * \brief   Check no model miss deadline, if so, terminate model
 * 
 * \endcond
 * ================================================================================================
 */
void 
Scheduler::missDeadlineHandler ()
{
#if (HARD_DEADLINE)
    for (auto app : mCPU->mAPPs)
    {
        list<Model*> missModels = {};
        auto model_info = app->modelInfo;
        
        /* check waiting model */
        for (auto model = app->waitingModels.begin(); model != app->waitingModels.end();)
        {
            if ((*model)->task.deadLine - model_info.totalExecuteTime <= total_gpu_cycle)
            {
                missModels.push_back(*model);
                app->waitingModels.erase(model++);
            } else {
                model++;
            }
        }

        /* check running model */
        for (auto model = app->runningModels.begin(); model != app->runningModels.end();)
        {
            int remaining_cycle = 0;
            auto kernel_status = (*model)->getKernelStatus();
            for (int i = 0; i < model_info.numOfLayers; i++) if(!kernel_status[i]) remaining_cycle += model_info.layerExecuteTime[i];
            
            if ((*model)->task.deadLine - remaining_cycle <= total_gpu_cycle)
            {
                missModels.push_back(*model);
                app->runningModels.erase(model++);
            } else {
                model++;
            }
        }

        /* handle miss deadline */
        for (auto model : missModels)
        {
            string buff = to_string(model->modelID) + " " + model->getModelName() + " with " + to_string(model->getBatchSize()) + " batch size miss deadline! [" + to_string(model->task.arrivalTime) + ", " + to_string(model->task.deadLine) + ", " + to_string(model->startTime) + ", " + to_string(total_gpu_cycle) + "]";
            log_E("Model", buff);

            ofstream file(LOG_OUT_PATH + program_name + ".txt", std::ios::app);
                file << "App " << model->appID << " Model " << buff << std::endl;
            file.close();

            model->memoryRelease(&mCPU->mMMU);

            mCPU->mGPU->terminateModel(model->appID, model->modelID);

            delete model;
        }
    }
#endif
}