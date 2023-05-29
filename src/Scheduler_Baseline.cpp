/**
 * \name    Approach_Baseline.cpp
 * 
 * \brief   Implement the callback function used in CPU.hpp.
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

    list<int> available_sm = mCPU->mGPU->getIdleSMs();
    if (available_sm.empty()) return false;
    
    /* *******************************************************************
     * Assign SM to each application according to its model needed pages
     * *******************************************************************
     */
    bool new_task = false;
    for (auto app : mCPU->mAPPs)
    {
        if(app->runningModels.empty() && !app->tasks.empty())
        {
            app->SM_budget = available_sm;
            new_task = true;
        }
    }
    if (!new_task)  return false;

#if (PRINT_SM_ALLCOATION_RESULT)
    for (auto app : mCPU->mAPPs)
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
    Model_Launcher();

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
bool
Scheduler::Model_Launcher ()
{
    bool new_model = false;
    for (auto app : mCPU->mAPPs)
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
            new_model = true;
        }
    }

    return new_model;
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
    list<Kernel*> readyKernels;
    for (auto app : mCPU->mAPPs)
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

        if (kernel->compileRequest(&mCPU->mMMU))
        {
            kernel->running = mCPU->mGPU->launchKernel(kernel);
            if (kernel->running) kernel->startCycle = total_gpu_cycle;
            
        } else {
            
            log_I("compileRequest", "kernel: " + to_string(kernel->kernelID) + "has empty requests");
        }
    }

    return true;
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

    /* check no application is running */
    if (command.INFERENCE_MODE == INFERENCE_TYPE::SEQUENTIAL)
    {
        for (auto app : mCPU->mAPPs) if (!app->runningModels.empty()) return false;
    }

    /*  Get avaiable SM list */
    list<int> available_sm = mCPU->mGPU->getIdleSMs();
    if (available_sm.size() < GPU_SM_NUM) return false;

    /* *******************************************************************
     * Assign SM to each application according to its model needed pages
     * *******************************************************************
     */
    bool new_task = false;
    for (auto app : mCPU->mAPPs)
    {
        if(!app->tasks.empty())
        {
            app->SM_budget = move(available_sm);
            new_task = true;
        }
    }
    if (!new_task)  return false;

#if (PRINT_SM_ALLCOATION_RESULT)
    for (auto app : mCPU->mAPPs)
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
    Model_Launcher();

    return true;
}