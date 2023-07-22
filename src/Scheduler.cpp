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
    /* *******************************************************************
     * Set Inference_Admission callback
     * *******************************************************************
     */
    if (command.SCHEDULER_MODE == SCHEDULER::Baseline)    Inference_Admission = Inference_Admission_API::Baseline;

    else if (command.SCHEDULER_MODE == SCHEDULER::Greedy) Inference_Admission = Inference_Admission_API::Greedy;

    else if (command.SCHEDULER_MODE == SCHEDULER::BARM)   Inference_Admission = Inference_Admission_API::BARM;

    else if (command.SCHEDULER_MODE == SCHEDULER::LazyB)  Inference_Admission = Inference_Admission_API::LazyB;

    else if (command.SCHEDULER_MODE == SCHEDULER::My)     Inference_Admission = Inference_Admission_API::My;

    else if (command.SCHEDULER_MODE == SCHEDULER::SALBI)  Inference_Admission = Inference_Admission_API::WA_SMD;

    /* *******************************************************************
     * Set Kernel_Scheduler callback
     * *******************************************************************
     */
    if (command.SCHEDULER_MODE == SCHEDULER::Baseline)    Kernel_Scheduler = Kernel_Scheduler_API::Baseline;

    else if (command.SCHEDULER_MODE == SCHEDULER::Greedy) Kernel_Scheduler = Kernel_Scheduler_API::Baseline;

    else if (command.SCHEDULER_MODE == SCHEDULER::BARM)   Kernel_Scheduler = Kernel_Scheduler_API::Baseline;

    else if (command.SCHEDULER_MODE == SCHEDULER::LazyB)  Kernel_Scheduler = Kernel_Scheduler_API::LazyB;

    else if (command.SCHEDULER_MODE == SCHEDULER::My)     Kernel_Scheduler = Kernel_Scheduler_API::My;

    else if (command.SCHEDULER_MODE == SCHEDULER::SALBI)  Kernel_Scheduler = Kernel_Scheduler_API::SALBI;


    /* *******************************************************************
     * Set Memory_Allocator callback
     * *******************************************************************
     */
    if (command.MEM_MODE == MEM_ALLOCATION::None)         Memory_Allocator = Memory_Allocator_API::None;

    else if (command.MEM_MODE == MEM_ALLOCATION::Average) Memory_Allocator = Memory_Allocator_API::Average;
    
    else if (command.MEM_MODE == MEM_ALLOCATION::MEMA)    Memory_Allocator = Memory_Allocator_API::MEMA;
    
    else if (command.MEM_MODE == MEM_ALLOCATION::R_MEMA)  Memory_Allocator = Memory_Allocator_API::R_MEMA;
    
    else if (command.MEM_MODE == MEM_ALLOCATION::BASLA)   Memory_Allocator = Memory_Allocator_API::BASLA;

    else ASSERT(false, "Set up Memory_Allocator callback error");
}


/** ===============================================================================================
 * \name    Sched
 * 
 * \brief   perform schedule
 * 
 * \endcond
 * ================================================================================================
 */
void 
Scheduler::Sched ()
{
    if (Inference_Admission (mCPU))

    Kernel_Scheduler    (mCPU);

    Memory_Allocator    (mCPU);
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