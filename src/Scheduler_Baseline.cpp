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
 * \name    Memory_Allocator
 * 
 * \brief   Allocate GPU unified memory (DRAM + VRAM) space to every application
 * 
 * \note    * This defualt scheduler didn't block SM to any application, just build tasks into model
 *          and launch all ready kernels to GPU's commandQueue. Which leads the kernel contension
 *          the SM resource in GPU core.
 * 
 * \endcond
 * ================================================================================================
 */
bool 
Scheduler::Memory_Allocator ()
{  
    if (command.MEM_MODE == MEM_ALLOCATION::None)
    {
        mCPU->mGPU->getGMMU()->setCGroupSize(-1, VRAM_SPACE / PAGE_SIZE);
        return true;
    }
    else if (command.MEM_MODE == MEM_ALLOCATION::Average)
    {
        map<int, int> memory_budget;

        int size = VRAM_SPACE / PAGE_SIZE;

        for (auto app : mCPU->mAPPs) memory_budget[app->appID] += floor(size / mCPU->mAPPs.size());

        for (int i = 0; i < size % mCPU->mAPPs.size(); i++) memory_budget[mCPU->mAPPs[i]->appID]++;
        
        for (auto app : mCPU->mAPPs) mCPU->mGPU->getGMMU()->setCGroupSize(app->appID, memory_budget[app->appID]);
        
        return true;
    }

    return false;
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
                file << "App " << model->appID << " Model " << buff << endl;
            file.close();

            model->memoryRelease(&mCPU->mMMU);

            mCPU->mGPU->terminateModel(model->appID, model->modelID);

            delete model;
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
    
    /* *******************************************************************
     * Check the models haven't miss deadline, if so, terminate model
     * *******************************************************************
     */
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
        while (!app->waitingModels.empty())
        {
            auto model = app->waitingModels.front();
            model->SM_budget = move(available_sm);

#if (PRINT_SM_ALLCOATION_RESULT)
            std::cout << "APP: " << app->appID << " Model: " << model->modelID << " get SM: ";
            for (auto sm_id : model->SM_budget) std::cout << sm_id << ", ";
            std::cout << std::endl;
#endif

            app->runningModels.push_back(model);
            app->waitingModels.pop_front();
        }
        
        if (!app->runningModels.empty()) break;
    }

    return true;
}