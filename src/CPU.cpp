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
    if (command.SM_MODE == SCHEDULER::LazyB)
    {
        mScheduler = new Scheduler_LazyB ( this );
    } 

    else if (command.INFERENCE_MODE == INFERENCE_TYPE::SEQUENTIAL || command.SM_MODE == SCHEDULER::Greedy) {
        mScheduler = new Scheduler_Greedy ( this );
    }

    else if (command.SM_MODE == SCHEDULER::Baseline) {
        mScheduler = new Scheduler_Baseline ( this );
    }

    else if (command.SM_MODE == SCHEDULER::BARM) {
        mScheduler = new Scheduler_BARM ( this );
    }

    /* *******************************************************************
     * Create applications
     * *******************************************************************
     */
    if (command.SM_MODE == SCHEDULER::LazyB)
    {
        int task_num = 16;
        mAPPs.push_back(new Application ((char*)"ResNet18" , {1, 3, 112, 112}, 1, 0, GPU_F / task_num, 0.1 * GPU_F, GPU_F));
        program_name = "LazyB_" + to_string(task_num) +"_ResNet18";
    }
    else if(command.TASK_MODE == TASK_SET::LIGHT)
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
        mAPPs.push_back(new Application ((char*)"LeNet"    , {1, 1, 32, 32}));
    }
    else if (command.TASK_MODE == TASK_SET::CaffeNet)
    {
        mAPPs.push_back(new Application ((char*)"CaffeNet" , {1, 3, 112, 112}));
    }
    else if (command.TASK_MODE == TASK_SET::ResNet18)
    {
        mAPPs.push_back(new Application ((char*)"ResNet18" , {1, 3, 112, 112}));
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

    mScheduler->Inference_Admission();

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
            string buff;

            if ((*model)->checkFinish())
            {
                buff = to_string((*model)->modelID) + " " + (*model)->getModelName() + " with " + to_string((*model)->getBatchSize()) + " batch size is finished [" + to_string((*model)->recorder.start_time) + ", " + to_string(total_gpu_cycle) + "]";
            } else if ((*model)->deadline < total_gpu_cycle) {
                buff = to_string((*model)->modelID) + " " + (*model)->getModelName() + " with " + to_string((*model)->getBatchSize()) + " batch size is miss deadline [" + to_string((*model)->recorder.start_time) + ", " + to_string(total_gpu_cycle) + "]";
            } else {
                model++;
                continue;
            }
             
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

