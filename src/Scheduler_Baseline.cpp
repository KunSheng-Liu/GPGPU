/**
 * \name    Scheduler_Baseline.cpp
 * 
 * \brief   Implement the basic function used in CPU.hpp.
 * 
 * \date    Jun 22, 2023
 */
#include "include/Scheduler.hpp"


/** ===============================================================================================
 * \name    Inference_Admission_API::Baseline
 * 
 * \brief   Allocate all SM to every application and its models
 * 
 * \note    * This defualt scheduler didn't block SM to any application, just build tasks into model
 *          and launch all ready kernels to GPU's commandQueue. Which leads the kernel contension
 *          the SM resource in GPU core.
 * 
 * \endcond
 * ================================================================================================
 */
bool 
Inference_Admission_API::Baseline (CPU* mCPU)
{  
    log_T("CPU", "Inference_Admission: Baseline");

    /* *******************************************************************
     * Assign SM to each application
     * *******************************************************************
     */
    unordered_set<int> available_sm;
    for (int i = 0; i < system_resource.SM_NUM; i++) available_sm.insert(i);
    
    for (auto app : mCPU->mAPPs)
    {
        while(!app->waitingModels.empty())
        {
            auto model = app->waitingModels.front();
            model->SM_budget = available_sm;

#if (PRINT_SM_ALLCOATION_RESULT)
            std::cout << "APP: " << app->appID << " Model: " << model->modelID << " get SM: ";
            for (auto sm_id : model->SM_budget) std::cout << sm_id << ", ";
            std::cout << std::endl;
#endif
            app->runningModels.push_back(model);
            app->waitingModels.pop_front();
        }
    }

    return true;
}


/** ===============================================================================================
 * \name    Inference_Admission_API::Greedy
 * 
 * \brief   Allocate all SM to an application, cannot release the SM until the model is totally 
 *          finish.
 * 
 * \endcond
 * ================================================================================================
 */
bool 
Inference_Admission_API::Greedy (CPU* mCPU)
{  
    log_T("CPU", "Inference_Admission: Greedy");

    /*  Get avaiable SM list */
    unordered_set<int> available_sm = mCPU->mGPU->getIdleSMs();
    if (available_sm.size() < system_resource.SM_NUM) return false;

    /* *******************************************************************
     * Assign SM to each application according to its model needed pages
     * *******************************************************************
     */
    for (auto app : mCPU->mAPPs)
    {
        while (!app->waitingModels.empty())
        {
            auto model = app->waitingModels.front();
            model->SM_budget = available_sm;

#if (PRINT_SM_ALLCOATION_RESULT)
            std::cout << "APP: " << app->appID << " Model: " << model->modelID << " get SM: ";
            for (auto sm_id : model->SM_budget) std::cout << sm_id << ", ";
            std::cout << std::endl;
#endif

            app->runningModels.push_back(model);
            app->waitingModels.pop_front();

            if (command.BATCH_MODE == BATCH_METHOD::DISABLE) break;
        }
        
        if (!app->runningModels.empty()) break;
    }

    return true;
}


/** ===============================================================================================
 * \name    Kernel_Scheduler_API::Baseline
 * 
 * \brief   Launch all ready kerenls to GPU command queue
 * 
 * \endcond
 * ================================================================================================
 */
bool
Kernel_Scheduler_API::Baseline (CPU* mCPU)
{  
    log_T("CPU", "Kernel_Scheduler: Baseline");

    /* *******************************************************************
     * Collect the ready kernels
     * *******************************************************************
     */
    for (auto app : mCPU->mAPPs)
    {
        list<Kernel*> readyKernels;
        for (auto model : app->runningModels)
        {
            for (auto kernel : model->findReadyKernels())
            {
                kernel->SM_List = &model->SM_budget;
                readyKernels.push_back(kernel);
            }
        }
        if (readyKernels.empty()) continue;
        
        /* print ready list */
#if (LOG_LEVEL >= VERBOSE)
        std::cout << "App " << app.appID << ": Ready kernel list: ";
        for (auto kernel : readyKernels)
        {
            std::cout << kernel->kernelID << ", ";
        }
        std::cout << std::endl;
#endif

        /* *******************************************************************
        * Launch kernel to gpu
        * *******************************************************************
        */
        Kernel* kernel = readyKernels.front();

        if (command.BATCH_MODE == BATCH_METHOD::MAX)
        {
            /* *******************************************************************
            * Choose kernel
            * *******************************************************************
            */
            vector<pair<Kernel*, int>> sync_kernels;
            unordered_set<int>* avaiable_sm = new unordered_set<int>;
            for (auto k : readyKernels)
            {
                if (k->srcLayer->layerID == kernel->srcLayer->layerID) 
                {
                    sync_kernels.push_back(make_pair(k, 1));
                    avaiable_sm->insert(k->SM_List->begin(), k->SM_List->end());
                }
            }
            if (sync_kernels.size() > 1) 
            {
                kernel = new KernelGroup(sync_kernels);
                kernel->SM_List = move(avaiable_sm);
            }
        }
        
        if (!kernel->SM_List->empty())
        {
            if (kernel->compileRequest(&mCPU->mMMU))
            {
                ASSERT(mCPU->mGPU->launchKernel(kernel), "Failed launch kernel");
                kernel->startCycle = total_gpu_cycle;
                kernel->running    = true;
            } 
            else log_I("compileRequest", "kernel: " + to_string(kernel->kernelID) + "has empty requests");
        }
    }

    
    return true;
}