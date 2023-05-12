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

#if (TASK_SET == TEST)
    mAPPs.push_back(new Application ((char*)"Test"));
    mAPPs.push_back(new Application ((char*)"VGG16"));
    mAPPs.push_back(new Application ((char*)"ResNet18"));
#elif (TASK_SET == LIGHT)
#elif (TASK_SET == HEAVY)
#elif (TASK_SET == MIX)
#endif

    
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

    Kernek_Inference_Scheduler();
    
    Dynamic_Batch_Admission();

    /* check new task */
    for (auto& app: mAPPs)
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

    /* *******************************************************************
     * Assign SM to each application according to its model needed pages
     * *******************************************************************
     */
#if (INFERENCE_METHOD == SEQUENTIAL)

    /* Only one application can get SM resource allocation when the GPU is idle */
    for (auto& app : mAPPs)
    {
        app->SM_budget = {};
        if (mGPU->idle() && !app->tasks.empty())
        {
            for (int i = 0; i < GPU_SM_NUM; i++)
            {
                app->SM_budget.push_back(i);
            }
            break;
        }
    }
    
#elif (INFERENCE_METHOD == PARALLEL)

    if (SM_MODE == SM_Dispatch::Baseline)
    {
        /* Each application got the same SM resource */
        for (auto& app : mAPPs)
        {
            app->SM_budget = {};

            if(app->tasks.size() == 0) continue;

            for (int i = 0; i < GPU_SM_NUM; i++)
            {
                app->SM_budget.push_back(i);
            }
        }
    } else if (SM_MODE == SM_Dispatch::SMD) 
    {
        /* Record the total required memory base on the task number */
        float total_needed_memory = 0;
        vector<pair<float, Application*>> APP_list;
        for (auto& app : mAPPs)
        {
            app->SM_budget = {};

            if(app->tasks.size() == 0) continue;

            auto info = app->modelInfo;

            total_needed_memory += (info.filterMemCount + info.filterMemCount) * app->tasks.size();  
            
            APP_list.emplace_back(make_pair((info.filterMemCount + info.filterMemCount) * app->tasks.size(), app));     
        }

        /* Sort to non-decreacing order */
        sort(APP_list.begin(), APP_list.end(), [](const pair<float, Application*>& a, const pair<float, Application*>& b){
            return a.first < b.first;
        });

        /* Assign SM to each application */
        int SM_count = 0;
        for (auto& app_pair : APP_list)
        {
            std::cout << GPU_SM_NUM * (app_pair.first / total_needed_memory) << std::endl;
            /* Avoid starvation, at least assign 1 SM to application */
            if ((int)(GPU_SM_NUM * (app_pair.first / total_needed_memory) == 0))
            {
                total_needed_memory -= app_pair.first;
                app_pair.second->SM_budget.push_back(SM_count++);
                continue;
            }

            for (int i = 0; i < (int)(GPU_SM_NUM * (app_pair.first / total_needed_memory)); i++)
            {
                app_pair.second->SM_budget.push_back(SM_count++);
            }

            ASSERT(SM_count == GPU_SM_NUM);
        }
    }

#endif

    /* Print SM allocation result */
#if (LOG_LEVEL >= VERBOSE)
    for (auto& app : mAPPs)
    {
        std::cout << "APP: " << app->appID << " get SM: ";
        for (auto sm_id : app->SM_budget)
        {
            std::cout << sm_id << ", ";
        }
        std::cout << std::endl;
    }
#endif
    /* Check allocation correct */
    // ASSERT(SM_count == GPU_SM_NUM);

    /* *******************************************************************
     * Choose the batch size of each model and create the instance
     * *******************************************************************
     */
    for (auto& app : mAPPs)
    {
        if (app->runningModels.empty() && !app->tasks.empty() && !app->SM_budget.empty())
        {
            int batchSize = 0;
#if (BATCH_INFERENCE)
            /* Determine the batch inference size */
            batchSize = app->tasks.size();
#else
            batchSize = 1;
#endif
            app->runningModels.emplace_back(new Model(app->appID, batchSize));
                
            Model* model = app->runningModels.back();

            model->record.SM_List = move(app->SM_budget);

            model->buildLayerGraph(app->modelType);
            model->memoryAllocate(&mMMU);

            /* If using real data, here should pass the data into IFMap */
            // auto batchInput = model->getIFMap();
            for (int i = 0; i < app->tasks.size(); i++)
            {
                Application::Task task = app->tasks.front();
                app->tasks.pop();

                // copy(task.data.begin(), task.data.end(), batchInput->begin() + i * task.data.size());
            }

        }
    }
}


/** ===============================================================================================
 * \name    Kernek_Inference_Scheduler
 * 
 * \brief   Second method, launch the model's kernel by the resource constrain.
 * 
 * \endcond
 * ================================================================================================
 */

void
CPU::Kernek_Inference_Scheduler()
{
    log_T("CPU", "Kernek_Inference_Scheduler");

    // handle the kernel dependency, and launch next kernel

    list<Kernel*> readyKernels;
    for (auto& app : mAPPs)
    {
        for (auto& model : app->runningModels)
        {
            for (auto& kernel : model->findReadyKernels())
            {
                kernel->record = &model->record;
                readyKernels.push_back(kernel);
            }
            
        }
    }

    /* print ready list */
#if (LOG_LEVEL >= VERBOSE)
    std::cout << "Ready kernel list: ";
    for (auto& kernel : readyKernels)
    {
        std::cout << kernel->kernelID << ", ";
    }
    std::cout << endl;
#endif
    /* launch kernel into gpu */
    for (auto& kernel : readyKernels)
    {
        ASSERT(kernel->record->SM_List.size());

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
    /* check finish kernel */
    if (!mGPU->finishedKernels.empty())
    {
        auto kernel = mGPU->finishedKernels.front();
        mGPU->finishedKernels.pop_front();

        log_W("Kernel", to_string(kernel->kernelID) + " is finished");
        kernel->finish = true;
        kernel->running = false;

        /* recording... */

        /* release */
        kernel->release();
        
    }

    for (auto& app : mAPPs)
    {
        for (auto model = app->runningModels.begin(); model != app->runningModels.end(); ++model) {
            if ((*model)->checkFinish())
            {
                log_W("Model", to_string((*model)->modelID) + " is finished");
                delete *model;
                model = app->runningModels.erase(model);
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
    for (auto& app : mAPPs)
    {
        finish &= app->finish;
    }
    return finish;
}
