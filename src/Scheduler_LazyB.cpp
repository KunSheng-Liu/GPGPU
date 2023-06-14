/**
 * \name    Approach_LazyB.cpp
 * 
 * \brief   Implement the function of related work \b Lazy_Batching used in CPU.hpp.
 * 
 * \details As a cloud server, the tasks is launched form the edge devices. Therefore try to maximize the batch size of the task 
 *          in kernel level.
 * 
 * \note    In this scenario, no memory limitation to the system.
 * \note    This approach use ResNet model
 * \note    Max batch size is constrained as 64
 * \note    The memory access overhead is set as 100 cycle
 * 
 * \date    May 25, 2023
 */
#include "include/Scheduler.hpp"


/** ===============================================================================================
 * \name    LazyB_Inference_Admission
 * 
 * \brief   Launch the tasks that can be merge into the model without violate the deadline.
 * 
 * \note    cannot terminate model when it's kernel is already running due to the sharing filter
 * 
 * \endcond
 * ================================================================================================
 */
bool 
Scheduler_LazyB::Inference_Admission ()
{  
    log_T("CPU", "Inference_Admission: LazyB");

    /* *******************************************************************
     * Check the models haven't miss deadline, if so, terminate model
     * *******************************************************************
     */
    missDeadlineHandler ();

    /* *******************************************************************
     * Allocate SM to application
     * *******************************************************************
     */
    vector<Application*> app_list;
    for (auto app : mCPU->mAPPs) if(!app->finish) app_list.push_back(app);
    if (app_list.empty()) return false;

    int sm_count = 0, sm_budget = system_resource.SM_NUM;
    while (sm_budget != 0) 
    {
        for (auto app : app_list) 
        {
            app->SM_budget.insert(sm_count++);
            if (!(--sm_budget)) break;
        }
    }

    /* *******************************************************************
     * Check the remaining slack time, and allocate SM
     * *******************************************************************
     */
    for (auto app : app_list)
    {
        if (!app->waitingModels.empty())
        {
            int batch_budget = LAZYB_MAX_BATCH_SIZE;
            double slack_time = app->arrivalTime - total_gpu_cycle;

            for (auto model : app->runningModels)
            {
                auto kernel_status = model->getKernelStatus();
                for (int i = 0; i < model->getNumOfLayer(); i++) if (!kernel_status[i]) slack_time -= model->getBatchSize() * app->modelInfo.layerExecuteTime[i];
            }

            int new_model = min(floor(slack_time / app->modelInfo.totalExecuteTime), (double)app->waitingModels.size());
            for (int i = 0; i < new_model; i++)
            {
                auto model = app->waitingModels.front();
                model->SM_budget = app->SM_budget;

#if (PRINT_SM_ALLCOATION_RESULT)
                std::cout << "APP: " << app->appID << " Model: " << model->modelID << " get SM: ";
                for (auto sm_id : model->SM_budget) std::cout << sm_id << ", ";
                std::cout << std::endl;
#endif
                app->runningModels.push_front(model);
                app->waitingModels.pop_front();
            }
        }
    }

    return true;
}


/** ===============================================================================================
 * \name    LazyB_Kernel_Scheduler
 * 
 * \brief   Launch the smallest ready kernel of all the models, if multiple model has smallest ready 
 *          kernels merge the models.
 * 
 * \endcond
 * ================================================================================================
 */
bool 
Scheduler_LazyB::Kernel_Scheduler ()
{  
    log_T("CPU", "Kernel_Scheduler: LazyB");

    /* *******************************************************************
     * Get Idle SMs
     * *******************************************************************
     */
    unordered_set<int> available_sm = mCPU->mGPU->getIdleSMs();

    for (auto app : mCPU->mAPPs)
    {
        /* *******************************************************************
         * Check the SM used for application is Idle
         * *******************************************************************
         */
        bool isReady = true;
        for (auto sm_id : app->SM_budget) isReady &= (available_sm.find(sm_id) != available_sm.end());
        if (!isReady || app->runningModels.empty()) continue;

#if (PRINT_LAZY_BATCHING)
        for (auto model : app->runningModels)
        {
            std::cout << "Model " << model->modelID << " with " << model->getBatchSize() << " batch size: Ready kernel list: ";
            for (auto kernel : model->findReadyKernels()) std::cout << kernel->srcLayer->layerID << ", ";
            std::cout << std::endl;
        }
#endif
        
        /* *******************************************************************
         * Perform merge
         * *******************************************************************
         */
        vector<pair<Kernel*, int>> sync_kernels;

        int latest_layer_id = app->runningModels.front()->findReadyKernels().front()->srcLayer->layerID;
        for (auto model : app->runningModels)
        {
            auto kernel = model->findReadyKernels().front();
            if (kernel->srcLayer->layerID == latest_layer_id)
            {
                sync_kernels.push_back(make_pair(kernel, model->getBatchSize()));
            }
        }

        /* *******************************************************************
         * Launch kernel to GPU
         * *******************************************************************
         */
        Kernel* kernel = (sync_kernels.size() == 1) ? sync_kernels.front().first : new KernelGroup(sync_kernels);
        
        kernel->SM_List = move(new unordered_set<int> (app->SM_budget));
        ASSERT(!kernel->SM_List->empty());

        if (kernel->compileRequest(&mCPU->mMMU))
        {
            ASSERT(mCPU->mGPU->launchKernel(kernel), "Failed launch kernel");
            kernel->startCycle = total_gpu_cycle;
            kernel->running    = true;
            
        } else {
            
            log_I("compileRequest", "kernel: " + to_string(kernel->kernelID) + "has empty requests");
        }
        
    }
    return false;
}