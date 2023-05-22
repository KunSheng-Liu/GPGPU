/**
 * \name    CPU.cpp
 * 
 * \brief   Implement the CPU and it's components.
 * 
 * \date    APR 6, 2023
 */

#include "include/CPU.hpp"

/** ===============================================================================================
 * \name    CPU
 * 
 * \brief   The class of the CPU, contains MMU, TLB.
 * 
 * \param   mc      the pointer of memory controller
 * \param   gpu     the pointer of GPU
 * 
 * \endcond
 * ================================================================================================
 */
CPU::CPU(MemoryController* mc, GPU* gpu) : mMC(mc), mGPU(gpu), mMMU(MMU(mc))
{
    if (command.TASK_MODE == TASK_SET::LIGHT)
    {
        mAPPs.push_back(new Application ((char*)"LeNet"    , {1, 1, 32, 32}));
        mAPPs.push_back(new Application ((char*)"ResNet18" , {1, 3, 112, 112}));
    }
    else if (command.TASK_MODE == TASK_SET::HEAVY)
    {
        mAPPs.push_back(new Application ((char*)"VGG16"    , {1, 3, 112, 112}));
        mAPPs.push_back(new Application ((char*)"GoogleNet", {1, 3, 112, 112}));
    }
    else if (command.TASK_MODE == TASK_SET::MIX)
    {
        mAPPs.push_back(new Application ((char*)"LeNet"    , {1, 1, 32, 32}));
        mAPPs.push_back(new Application ((char*)"ResNet18" , {1, 3, 112, 112}));
        mAPPs.push_back(new Application ((char*)"GoogleNet", {1, 3, 112, 112}));
        mAPPs.push_back(new Application ((char*)"VGG16"    , {1, 3, 112, 112}));
    }
    else if (command.TASK_MODE == TASK_SET::ALL)
    {
        mAPPs.push_back(new Application ((char*)"LeNet"    , {1, 1, 32, 32}));
        mAPPs.push_back(new Application ((char*)"ResNet18" , {1, 3, 112, 112}));
        mAPPs.push_back(new Application ((char*)"CaffeNet" , {1, 3, 112, 112}));
        mAPPs.push_back(new Application ((char*)"GoogleNet", {1, 3, 112, 112}));
        mAPPs.push_back(new Application ((char*)"VGG16"    , {1, 3, 112, 112}));
    }
    else if (command.TASK_MODE == TASK_SET::LeNet)
    {
        mAPPs.push_back(new Application ((char*)"LeNet"    , {1, 3, 112, 112}));
    }
    else if (command.TASK_MODE == TASK_SET::CaffeNet)
    {
        mAPPs.push_back(new Application ((char*)"CaffeNet" , {1, 3, 112, 112}));
    }
    else if (command.TASK_MODE == TASK_SET::ResNet18)
    {
        mAPPs.push_back(new Application ((char*)"CaffeNet" , {1, 3, 112, 112}));
    }
    else if (command.TASK_MODE == TASK_SET::VGG16)
    {
        mAPPs.push_back(new Application ((char*)"VGG16"    , {1, 3, 112, 112}));
    }
    else if (command.TASK_MODE == TASK_SET::GoogleNet)
    {
        mAPPs.push_back(new Application ((char*)"GoogleNet", {1, 3, 112, 112}));
    }
    else if (command.TASK_MODE == TASK_SET::TEST1)
    {
        mAPPs.push_back(new Application ((char*)"ResNet18" , {1, 3, 112, 112}));
        mAPPs.push_back(new Application ((char*)"VGG16"    , {1, 3, 112, 112}));
        mAPPs.push_back(new Application ((char*)"GoogleNet", {1, 3, 112, 112}));
    }
    else if (command.TASK_MODE == TASK_SET::TEST2)
    {
        mAPPs.push_back(new Application ((char*)"ResNet18" , {1, 3, 112, 112}, 3));
        mAPPs.push_back(new Application ((char*)"VGG16"    , {1, 3, 112, 112}, 1));
        mAPPs.push_back(new Application ((char*)"GoogleNet", {1, 3, 112, 112}, 2));
    }
    else {
        ASSERT(false, "Test set error");
    }
}


/** ===============================================================================================
 * \name    CPU
 * 
 * \brief   Destruct CPU
 * 
 * \endcond
 * ================================================================================================
 */
CPU::~CPU()
{
    for (auto app = mAPPs.begin(); app != mAPPs.end(); ++app) {
        delete *app;
    }
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
CPU::cycle()
{
    log_I("CPU Cycle", to_string(total_gpu_cycle));

    Check_Finish_Kernel();
    
    Dynamic_Batch_Admission();

    Kernel_Inference_Scheduler();

    /* check new task */
    for (auto app: mAPPs)
    {
        app->cycle();
    }
}


/** ===============================================================================================
 * \name    Dynamic_Batching_Admission
 * 
 * \brief   First method, to determine the batch size of each model by the given information.
 * 
 * \endcond
 * ================================================================================================
 */

void
CPU::Dynamic_Batch_Admission()
{
    log_T("CPU", "Dynamic_Batching_Algorithm");

    /*  Get avaiable SM list */
    list<int> available_sm = mGPU->getIdleSMs();
    if (available_sm.empty()) return;

    /* *******************************************************************
     * Assign SM to each application according to its model needed pages
     * *******************************************************************
     */
    if (command.INFERENCE_MODE == INFERENCE_TYPE::SEQUENTIAL)
    {
        if (!SM_Greedy_Scheduler()) return;

    } 
    else if (command.INFERENCE_MODE == INFERENCE_TYPE::PARALLEL)
    {
        if (command.SM_MODE == SM_DISPATCH::Greedy)
        {
            if (!SM_Greedy_Scheduler()) return;
        }
        else if (command.SM_MODE == SM_DISPATCH::Baseline)
        {
            /* Each application got the same SM resource */
            if (!SM_Baseline_Scheduler()) return;
        } 
        else if (command.SM_MODE == SM_DISPATCH::SMD) 
        {
            SM_SMD_Scheduler();
        } else {
            ASSERT(false, "SM dispatch error");
        }

    } else {
        ASSERT(false, "Inference method error");
    }

#if (PRINT_SM_ALLCOATION_RESULT)
    for (auto app : mAPPs)
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
    for (auto app : mAPPs)
    {
        /* Each application can only run one model in the same time */
        if (app->runningModels.empty() && !app->SM_budget.empty() && !app->tasks.empty())
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
}


/** ===============================================================================================
 * \name    Kernel_Inference_Scheduler
 * 
 * \brief   Second method, launch the model's kernel by the resource constrain.
 * 
 * \endcond
 * ================================================================================================
 */

void
CPU::Kernel_Inference_Scheduler()
{
    log_T("CPU", "Kernek_Inference_Scheduler");

    // handle the kernel dependency, and launch next kernel

    list<Kernel*> readyKernels;
    for (auto app : mAPPs)
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
    /* launch kernel into gpu */
    for (auto kernel : readyKernels)
    {
        ASSERT(kernel->SM_List->size());

        if (kernel->compileRequest(&mMMU))
        {
            kernel->running = mGPU->launchKernel(kernel);
            
        } else {
            
            log_I("compileRequest", "kernel: " + to_string(kernel->kernelID) + "has empty requests");
        }
    }    
}


/** ===============================================================================================
 * \name    Check_Finish_Kernel
 * 
 * \brief   Check whether the kernel has been finished, Record kernel inside finishedKernels.
 * 
 * \endcond
 * ================================================================================================
 */
void
CPU::Check_Finish_Kernel()
{  
    bool check_finish = !mGPU->finishedKernels.empty();

    /* check finish kernel */
    while (!mGPU->finishedKernels.empty())
    {
        auto kernel = mGPU->finishedKernels.front();
        mGPU->finishedKernels.pop_front();

        /* recording... */
#if (PRINT_BLOCK_RECORD)
            ofstream file(LOG_OUT_PATH + program_name + ".txt", std::ios::app);
            file << "Finish kernel" << std::right << setw(4) << kernel->kernelID << ":" << std::endl;
            for (auto& b_record : kernel->block_record)
            {
                file << "Finish block" << std::right << setw(5) << b_record.block_id << ": [" 
                     << b_record.sm_id                 << ", "
                     << b_record.start_cycle           << ", "
                     << b_record.end_cycle             << ", "
                     << b_record.launch_access_counter << ", "
                     << b_record.return_access_counter << ", "
                     << b_record.access_page_counter   << "]"
                     << std::endl;
    #if (PRINT_WARP_RECORD)
                for (auto& w_record : b_record.warp_record)
                {
                    file << std::right << setw(14) << "warp" << std::right << setw(3) << w_record.warp_id << ": ["
                         << w_record.start_cycle         << ", "
                         << w_record.end_cycle           << ", "
                         << w_record.computing_cycle     << ", "
                         << w_record.wait_cycle          << "]"
                         << std::endl;
                }
    #endif
            }
            file.close();
#endif

        kernel->finish = true;
        kernel->running = false;
        log_W("Kernel", to_string(kernel->kernelID) + " (" + kernel->srcLayer->layerType + ") is finished");
    }

    if (check_finish)
    {
        for (auto app : mAPPs)
        {
            for (auto model = app->runningModels.begin(); model != app->runningModels.end(); ++model) {
                if ((*model)->checkFinish())
                {
                    /* release the used resource */
                    (*model)->memoryRelease(&mMMU);
                    mGPU->getGMMU()->freeCGroup((*model)->modelID);

                    /* log out */
                    log_W("Model", to_string((*model)->modelID) + " " + (*model)->getModelName() + " with " + to_string((*model)->getBatchSize()) + " batch size is finished");
                    
                    ofstream file(LOG_OUT_PATH + program_name + ".txt", std::ios::app);
                        file << "App " << (*model)->appID << " Model " << (*model)->modelID << ": " << (*model)->getModelName() << " with " << (*model)->getBatchSize() << " batch size is finished" << endl;
                    file.close();

                    /* delete the model */
                    delete *model;
                    model = app->runningModels.erase(model);
                }
            }
        }
    }   
}


/** ===============================================================================================
 * \name    Check_All_Applications_Finish
 * 
 * \brief   Check whether the applications has finished.
 * 
 * \endcond
 * ================================================================================================
 */
bool
CPU::Check_All_Applications_Finish()
{  
    bool finish = true;
    for (auto app : mAPPs) finish &= app->finish;

    return finish;
}


/** ===============================================================================================
 * \name    SM_Greedy_Scheduler
 * 
 * \brief   Only one application can get SM resource when the GPU is totally idle
 * 
 * \endcond
 * ================================================================================================
 */
bool
CPU::SM_Greedy_Scheduler()
{  
    list<int> available_sm = mGPU->getIdleSMs();
    if (available_sm.size() == GPU_SM_NUM)
    {
        for (auto app : mAPPs) if (!available_sm.empty() && !app->finish) app->SM_budget = move(available_sm);
        return true;
    }
    return false;
}


/** ===============================================================================================
 * \name    SM_Baseline_Scheduler
 * 
 * \brief   Let the application run-time contend the SM resource
 * 
 * \endcond
 * ================================================================================================
 */
bool
CPU::SM_Baseline_Scheduler()
{  
    bool new_task = false;
    list<int> available_sm = mGPU->getIdleSMs();
    for (auto app : mAPPs)
    {
        if(app->tasks.size() == 0) continue;

        app->SM_budget = available_sm;
        new_task = true;
    }

    return new_task;
}


/** ===============================================================================================
 * \name    SM_SMD_Scheduler
 * 
 * \brief   Allocate SM to each application according to the memory usage
 * 
 * \endcond
 * ================================================================================================
 */
bool
CPU::SM_SMD_Scheduler()
{  
    list<int> available_sm = mGPU->getIdleSMs();

    ASSERT(false, "haven't implement SMD");

    // /* Record the total required memory base on the task number */
    // float total_needed_memory = 0;
    // vector<pair<float, Application*>> APP_list;
    // for (auto app : mAPPs)
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

