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
    /* *******************************************************************
     * Register the callback functions
     * *******************************************************************
     */
    if (command.SCHEDULER_MODE == SCHEDULER::LazyB)
    {
        mScheduler = new Scheduler_LazyB ( this );
    } 
    
    else if (command.SCHEDULER_MODE == SCHEDULER::My) {
        mScheduler = new Scheduler_My ( this );
    }

    else if (command.SCHEDULER_MODE == SCHEDULER::Greedy) {
        mScheduler = new Scheduler_Greedy ( this );
    }

    else if (command.SCHEDULER_MODE == SCHEDULER::Baseline) {
        mScheduler = new Scheduler_Baseline ( this );
    }

    else if (command.SCHEDULER_MODE == SCHEDULER::BARM) {
        mScheduler = new Scheduler_BARM ( this );
    }

    /* *******************************************************************
     * Create applications
     * *******************************************************************
     */
    for (auto& task : command.TASK_LIST)
    {
        if (task.first == APPLICATION::LeNet)
        {
            mAPPs.push_back(new Application ((char*)"LeNet"
                , {1, 1, 32, 32}
                , (command.SCHEDULER_MODE == SCHEDULER::LazyB) ? 1 : task.second, 0
                , GPU_F / (command.SCHEDULER_MODE == SCHEDULER::LazyB) ? GPU_F / task.second : -1
                , GPU_F * TASK_DEADLINE / 1000
                , GPU_F * SIMULATION_TIME / 1000
            ));
        }
        else if (task.first == APPLICATION::CaffeNet)
        {
            mAPPs.push_back(new Application ((char*)"CaffeNet"
                , {1, 1, 112, 112}
                , (command.SCHEDULER_MODE == SCHEDULER::LazyB) ? 1 : task.second, 0
                , GPU_F / (command.SCHEDULER_MODE == SCHEDULER::LazyB) ? GPU_F / task.second : -1
                , GPU_F * TASK_DEADLINE / 1000
                , GPU_F * SIMULATION_TIME / 1000
            ));
        }
        else if (task.first == APPLICATION::ResNet18)
        {
            mAPPs.push_back(new Application ((char*)"ResNet18"
                , {1, 1, 112, 112}
                , (command.SCHEDULER_MODE == SCHEDULER::LazyB) ? 1 : task.second, 0
                , GPU_F / (command.SCHEDULER_MODE == SCHEDULER::LazyB) ? GPU_F / task.second : -1
                , GPU_F * TASK_DEADLINE / 1000
                , GPU_F * SIMULATION_TIME / 1000
            ));
        }
        else if (task.first == APPLICATION::GoogleNet)
        {
            mAPPs.push_back(new Application ((char*)"GoogleNet"
                , {1, 1, 112, 112}
                , (command.SCHEDULER_MODE == SCHEDULER::LazyB) ? 1 : task.second, 0
                , GPU_F / (command.SCHEDULER_MODE == SCHEDULER::LazyB) ? GPU_F / task.second : -1
                , GPU_F * TASK_DEADLINE / 1000
                , GPU_F * SIMULATION_TIME / 1000
            ));
        }
        else if (task.first == APPLICATION::VGG16)
        {
            mAPPs.push_back(new Application ((char*)"VGG16"
                , {1, 1, 112, 112}
                , (command.SCHEDULER_MODE == SCHEDULER::LazyB) ? 1 : task.second, 0
                , GPU_F / (command.SCHEDULER_MODE == SCHEDULER::LazyB) ? GPU_F / task.second : -1
                , GPU_F * TASK_DEADLINE / 1000
                , GPU_F * SIMULATION_TIME / 1000
            ));
        }
        else if(task.first == APPLICATION::LIGHT)
        {
            command.TASK_LIST.emplace_back(make_pair(APPLICATION::LeNet,     1));
            command.TASK_LIST.emplace_back(make_pair(APPLICATION::ResNet18,  1));
        }
        else if (task.first == APPLICATION::HEAVY)
        {
            command.TASK_LIST.emplace_back(make_pair(APPLICATION::GoogleNet, 1));
            command.TASK_LIST.emplace_back(make_pair(APPLICATION::VGG16,     1));
        }
        else if (task.first == APPLICATION::MIX)
        {
            command.TASK_LIST.emplace_back(make_pair(APPLICATION::LeNet,     1));
            command.TASK_LIST.emplace_back(make_pair(APPLICATION::ResNet18,  1));
            command.TASK_LIST.emplace_back(make_pair(APPLICATION::GoogleNet, 1));
            command.TASK_LIST.emplace_back(make_pair(APPLICATION::VGG16,     1));
        }
        else if (task.first == APPLICATION::ALL)
        {
            command.TASK_LIST.emplace_back(make_pair(APPLICATION::LeNet,     1));
            command.TASK_LIST.emplace_back(make_pair(APPLICATION::ResNet18,  1));
            command.TASK_LIST.emplace_back(make_pair(APPLICATION::GoogleNet, 1));
            command.TASK_LIST.emplace_back(make_pair(APPLICATION::VGG16,     1));
        }
        else {
            ASSERT(false, "Test set error");
        }
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

    (Check_Finish_Kernel() || mGPU->isIdle()) &&
    
    mScheduler->Inference_Admission() && 
    
    mScheduler->Kernel_Scheduler();

    /* check new task */
    for (auto app: mAPPs)
    {
        app->cycle();
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
bool
CPU::Check_Finish_Kernel()
{  
    if (mGPU->finishedKernels.empty()) return false;

    /* check finish kernel */
    while (!mGPU->finishedKernels.empty())
    {
        auto kernel = mGPU->finishedKernels.front();
        mGPU->finishedKernels.pop_front();

        kernel->handleKernelCompletion ();
    }

    /* *******************************************************************
     * Check whether the model is finished
     * *******************************************************************
     */
    for (auto app : mAPPs)
    {
        for (auto model = app->runningModels.begin(); model != app->runningModels.end();) {

            if (!(*model)->checkFinish())
            {
                model++;
                continue;
            }
             
            string buff = to_string((*model)->modelID) + " " + (*model)->getModelName() + " with " + to_string((*model)->getBatchSize()) + " batch size is finished [" + to_string((*model)->task.arrivalTime) + ", " + to_string((*model)->task.deadLine) + ", " + to_string((*model)->startTime) + ", " + to_string(total_gpu_cycle) + "]";
            log_W("Model", buff);
            
            /* Release the used memory */
            PageRecord page_record = (*model)->memoryRelease(&mMMU);
            mGPU->getGMMU()->freeCGroup((*model)->modelID);

            /* Record the kernel information into file */
            ofstream file(LOG_OUT_PATH + program_name + ".txt", std::ios::app);
                file << "PageRecord: [" << page_record.read_counter << ", " << page_record.write_counter << ", " << page_record.access_count << ", " << page_record.swap_count << "]" << endl;
                file << "App " << (*model)->appID << " Model " << buff << endl;
            file.close();

            /* Delete the model */
            delete *model;
            app->runningModels.erase(model++);
        }
    }

    return true;
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

