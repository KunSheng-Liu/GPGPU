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
    if (command.INFERENCE_MODE == INFERENCE_TYPE::SEQUENTIAL || command.SM_MODE == SM_DISPATCH::Greedy)
    {
        Inference_Admission = &Greedy_Inference_Admission;
    } 

    else if (command.SM_MODE == SM_DISPATCH::Baseline) {
        Inference_Admission = &Baseline_Inference_Admission;
    }

    else if (command.SM_MODE == SM_DISPATCH::SMD) {
        Inference_Admission = &BARM_Inference_Admission;
    }

    Kernel_Scheduler = &Baseline_Kernel_Scheduler;

    /* *******************************************************************
     * Create applications
     * *******************************************************************
     */
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
    
    Inference_Admission (this);

    Kernel_Scheduler (this);

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
void
CPU::Check_Finish_Kernel()
{  
    if (mGPU->finishedKernels.empty()) return;

    /* check finish kernel */
    while (!mGPU->finishedKernels.empty())
    {
        auto kernel = mGPU->finishedKernels.front();
        mGPU->finishedKernels.pop_front();

        kernel->finish = true;
        kernel->running = false;
        log_W("Kernel", to_string(kernel->kernelID) + " (" + kernel->srcLayer->layerType + ") is finished");

        /* *******************************************************************
         * Record the kernel information into file
         * *******************************************************************
         */
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
    }

    /* *******************************************************************
     * Check whether the model is finished
     * *******************************************************************
     */
    for (auto app : mAPPs)
    {
        for (auto model = app->runningModels.begin(); model != app->runningModels.end(); ++model) {
            if ((*model)->checkFinish())
            {
                string buff = to_string((*model)->modelID) + " " + (*model)->getModelName() + " with " + to_string((*model)->getBatchSize()) + " batch size is finished [" + to_string((*model)->recorder.start_time) + ", " + to_string(total_gpu_cycle) + "]";
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
    for (auto app : mAPPs) finish &= app->finish;

    return finish;
}

