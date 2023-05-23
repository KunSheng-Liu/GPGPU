/**
 * \name    Approach_Baseline.cpp
 * 
 * \brief   Implement the callback function used in CPU.hpp.
 * 
 * \date    May 23, 2023
 */
#include "include/Approach.hpp"

/** ===============================================================================================
 * \name    Greedy_Inference_Admission
 * 
 * \brief   Allocate all SM to an application, cannot release the SM until the model is totally 
 *          finish.
 * 
 * \endcond
 * ================================================================================================
 */
bool Greedy_Inference_Admission (CPU* cpu)
{  
    log_T("CPU", "Inference_Admission: Greedy");

    /* check no application is running */
    if (command.INFERENCE_MODE == INFERENCE_TYPE::SEQUENTIAL)
    {
        for (auto app : cpu->mAPPs) if (!app->runningModels.empty()) return false;
    }

    /*  Get avaiable SM list */
    list<int> available_sm = cpu->mGPU->getIdleSMs();
    if (available_sm.size() < GPU_SM_NUM) return false;

    /* *******************************************************************
     * Assign SM to each application according to its model needed pages
     * *******************************************************************
     */
    bool new_task = false;
    for (auto app : cpu->mAPPs)
    {
        if(!app->tasks.empty())
        {
            app->SM_budget = move(available_sm);
            new_task = true;
        }
    }
    if (!new_task)  return false;

#if (PRINT_SM_ALLCOATION_RESULT)
    for (auto app : cpu->mAPPs)
    {
        std::cout << "APP: " << app->appID << " get SM: ";
        for (auto sm_id : app->SM_budget) std::cout << sm_id << ", ";
        std::cout << std::endl;
    }
#endif

    /* *******************************************************************
     * Choose the batch size of each model and create the instance
     * *******************************************************************
     */
    Default_Model_Launcher (cpu);

    return true;
}


/** ===============================================================================================
 * \name    Baseline_Inference_Admission
 * 
 * \brief   Allocate SM to every application, but each application can only execte one model in a 
 *          time.
 * 
 * \endcond
 * ================================================================================================
 */
bool Baseline_Inference_Admission (CPU* cpu)
{  
    log_T("CPU", "Inference_Admission: Baseline");

    list<int> available_sm = cpu->mGPU->getIdleSMs();
    if (available_sm.empty()) return false;
    
    /* *******************************************************************
     * Assign SM to each application according to its model needed pages
     * *******************************************************************
     */
    bool new_task = false;
    for (auto app : cpu->mAPPs)
    {
        if(app->runningModels.empty() && !app->tasks.empty())
        {
            app->SM_budget = available_sm;
            new_task = true;
        }
    }
    if (!new_task)  return false;

#if (PRINT_SM_ALLCOATION_RESULT)
    for (auto app : cpu->mAPPs)
    {
        std::cout << "APP: " << app->appID << " get SM: ";
        for (auto sm_id : app->SM_budget) std::cout << sm_id << ", ";
        std::cout << std::endl;
    }
#endif

    /* *******************************************************************
     * Choose the batch size of each model and create the instance
     * *******************************************************************
     */
    Default_Model_Launcher(cpu);

    return true;
}


/** ===============================================================================================
 * \name    Default_Model_Launcher
 * 
 * \brief   Launch model if the application get the SM resource, choose the batch size according to
 *          the configuration.
 * 
 * \endcond
 * ================================================================================================
 */
bool Default_Model_Launcher (CPU* cpu)
{  
    for (auto app : cpu->mAPPs)
    {
        /* Each application can only run one model in the same time */
        if (!app->SM_budget.empty() && !app->tasks.empty())
        {
            int batchSize;
            if (command.BATCH_MODE == BATCH_METHOD::DISABLE)
            {
                batchSize = 1;
            }
            else if (command.BATCH_MODE == BATCH_METHOD::MAX)
            {
                batchSize = app->tasks.size();
            }

            app->runningModels.emplace_back(new Model(app->appID, app->modelType, app->inputSize, batchSize));
                
            Model* model = app->runningModels.back();

            model->SM_budget = move(app->SM_budget);

            model->buildLayerGraph();

            /* If using real data, here should pass the data into IFMap */
            // auto batchInput = model->getIFMap();
            for (int i = 0; i < batchSize; i++)
            {
                Application::Task task = app->tasks.front();
                app->tasks.pop();

                // copy(task.data.begin(), task.data.end(), batchInput->begin() + i * task.data.size());
            }

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
bool Baseline_Kernel_Scheduler (CPU* cpu)
{  
    log_T("CPU", "Kernel_Scheduler: Baseline");

    // handle the kernel dependency, and launch next kernel
    list<Kernel*> readyKernels;
    for (auto app : cpu->mAPPs)
    {
        for (auto model : app->runningModels)
        {
            for (auto kernel : model->findReadyKernels())
            {
                kernel->SM_List = &model->SM_budget;
                readyKernels.push_back(kernel);
            }
            
        }
    }

    /* print ready list */
#if (LOG_LEVEL >= VERBOSE)
    std::cout << "Ready kernel list: ";
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
    for (auto kernel : readyKernels)
    {
        ASSERT(!kernel->SM_List->empty());

        if (kernel->compileRequest(&cpu->mMMU))
        {
            kernel->running = cpu->mGPU->launchKernel(kernel);
            
        } else {
            
            log_I("compileRequest", "kernel: " + to_string(kernel->kernelID) + "has empty requests");
        }
    }

    return true;
}


/** ===============================================================================================
 * \name    BARM_Inference_Admission
 * 
 * \brief   Use BARM::SMD scheme to allocate SM
 * 
 * \endcond
 * ================================================================================================
 */
bool BARM_Inference_Admission (CPU* cpu)
{  
    log_T("CPU", "Dynamic_Batching_Algorithm: SM_SMD_Scheduler");

    /*  Get avaiable SM list */
    list<int> available_sm = cpu->mGPU->getIdleSMs();
    if (available_sm.empty()) return false;

    ASSERT(false, "haven't implement SMD");

    /* *******************************************************************
     * Assign SM to each application according to its model needed pages
     * *******************************************************************
     */
    /* Record the total required memory base on the task number */
    // float total_needed_memory = 0;
    // vector<pair<float, Application*>> APP_list;
    // for (auto app : cpu->mAPPs)
    // {
    //     app->SM_budget = {};

    //     if(app->tasks.size() == 0) continue;

    //     auto info = app->modelInfo;

    //     total_needed_memory += (info.filterMemCount + info.filterMemCount) * app->tasks.size();  
        
    //     APP_list.emplace_back(make_pair((info.filterMemCount + info.filterMemCount) * app->tasks.size(), app));     
    // }

    // /* Sort to non-decreacing order */
    // sort(APP_list.begin(), APP_list.end(), [](const pair<float, Application*>& a, const pair<float, Application*>& b){
    //     return a.first < b.first;
    // });

    // /* Assign SM to each application */
    // int SM_count = 0;
    // for (auto app_pair : APP_list)
    // {
    //     std::cout << GPU_SM_NUM * (app_pair.first / total_needed_memory) << std::endl;
    //     /* Avoid starvation, at least assign 1 SM to application */
    //     if ((int)(GPU_SM_NUM * (app_pair.first / total_needed_memory) == 0))
    //     {
    //         total_needed_memory -= app_pair.first;
    //         app_pair.second->SM_budget.push_back(SM_count++);
    //         continue;
    //     }

    //     for (int i = 0; i < (int)(GPU_SM_NUM * (app_pair.first / total_needed_memory)); i++)
    //     {
    //         app_pair.second->SM_budget.push_back(SM_count++);
    //     }

    //     ASSERT(SM_count == GPU_SM_NUM);
    // }

}

/** ===============================================================================================
 * \name    Lazy_Batching_Kernel_Scheduler
 * 
 * \brief   Only one application can get SM resource when the GPU is totally idle
 * 
 * \endcond
 * ================================================================================================
 */
bool Lazy_Batching_Kernel_Scheduler (CPU* cpu)
{  
    log_T("CPU", "Dynamic_Batching_Algorithm: SM_SMD_Scheduler");

    /*  Get avaiable SM list */
    list<int> available_sm = cpu->mGPU->getIdleSMs();
    if (available_sm.empty()) return false;

    /* *******************************************************************
     * Assign SM to each application according to its model needed pages
     * *******************************************************************
     */
    // ...
    ASSERT(false, "haven't implement Lazy Batching");

    return false;
}