/**
 * \name    Approach_Baseline.cpp
 * 
 * \brief   Implement the basic function used in CPU.hpp.
 * 
 * \date    May 23, 2023
 */
#include "include/Scheduler.hpp"


/** ===============================================================================================
 * \name    Baseline_Inference_Admission
 * 
 * \brief   Allocate SM to every application, but each application can only execte one model in a 
 *          time.
 * 
 * \note    * This defualt scheduler didn't block SM to any application, just build tasks into model
 *          and launch all ready kernels to GPU's commandQueue. Which leads the kernel contension
 *          the SM resource in GPU core.
 * 
 * \note    * All application can run one model once a time
 * 
 * \endcond
 * ================================================================================================
 */
bool 
Scheduler::Inference_Admission ()
{  
    log_T("CPU", "Inference_Admission: Baseline");

    /* *******************************************************************
     * Check the models haven't miss deadline, if so, terminate model
     * *******************************************************************
     */
    missDeadlineHandler();

    /* *******************************************************************
     * Assign SM to each application
     * *******************************************************************
     */
    unordered_set<int> available_sm;
    for (int i = 0; i < GPU_SM_NUM; i++) available_sm.insert(i);
    
    for (auto app : mCPU->mAPPs)
    {
        for (auto rmodel = app->runningModels.rbegin(); rmodel != app->runningModels.rend(); rmodel++)
        {
            if ((*rmodel)->SM_budget.empty()) (*rmodel)->SM_budget = available_sm;
            else break;

#if (PRINT_SM_ALLCOATION_RESULT)
            std::cout << "APP: " << app->appID << " Model: " << (*rmodel)->modelID << " get SM: ";
            for (auto sm_id : (*rmodel)->SM_budget) std::cout << sm_id << ", ";
            std::cout << std::endl;
#endif
        }
    }

    return true;
}



/** ===============================================================================================
 * \name    Baseline_Kernel_Scheduler
 * 
 * \brief   Launch all ready kerenls to GPU command queue
 * 
 * \endcond
 * ================================================================================================
 */
bool
Scheduler::Kernel_Scheduler ()
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
        std::cout << endl;
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
                
            } else {

                log_I("compileRequest", "kernel: " + to_string(kernel->kernelID) + "has empty requests");
            }
        }
    }

    
    return true;
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
    for (auto app : mCPU->mAPPs)
    {
        for (auto model = app->runningModels.begin(); model != app->runningModels.end();)
        {
            if ((*model)->task.deadLine <= total_gpu_cycle)
            {
                std::cout << "Model " << (*model)->modelID << " miss deadline!" << std::endl;
                (*model)->memoryRelease(&mCPU->mMMU);

                mCPU->mGPU->terminateModel((*model)->modelID);

                delete *model;
                app->runningModels.erase(model++);
            } else {
                model++;
            }
        }
    }
}



/** ===============================================================================================
 * \name    Greedy_Inference_Admission
 * 
 * \brief   Allocate all SM to an application, cannot release the SM until the model is totally 
 *          finish.
 * 
 * \endcond
 * ================================================================================================
 */
bool 
Scheduler_Greedy::Inference_Admission ()
{  
    log_T("CPU", "Inference_Admission: Greedy");

    missDeadlineHandler();

    /*  Get avaiable SM list */
    unordered_set<int> available_sm = mCPU->mGPU->getIdleSMs();
    if (available_sm.size() < GPU_SM_NUM) return false;

    /* *******************************************************************
     * Assign SM to each application according to its model needed pages
     * *******************************************************************
     */
    for (auto app : mCPU->mAPPs)
    {
        for (auto model : app->runningModels)
        {
            model->SM_budget = move(available_sm);

#if (PRINT_SM_ALLCOATION_RESULT)
            std::cout << "APP: " << app->appID << " Model: " << model->modelID << " get SM: ";
            for (auto sm_id : model->SM_budget) std::cout << sm_id << ", ";
            std::cout << std::endl;
#endif
            return true;
        }
    }

    return false;
}