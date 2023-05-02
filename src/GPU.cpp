/**
 * \name    GPU.cpp
 * 
 * \brief   Implement the CPU.
 * 
 * \date    APR 30, 2023
 */

#include "include/GPU.hpp"

/** ===============================================================================================
 * \name    GPU
 * 
 * \brief   The class of the GPU
 * 
 * \param   mc      the pointer of memory controller
 * 
 * \endcond
 * ================================================================================================
 */
GPU::GPU(MemoryController* mc) : mMC(mc), mGMMU(GMMU()), mSMs(list<SM>(GPU_SM_NUM))
{
    /* Link the gmmu into each SM */
    for (auto& sm : mSMs)
    {
        sm.setGMMU(&mGMMU);
    }
}


/** ===============================================================================================
 * \name    GPU
 * 
 * \brief   Destruct GPU
 * 
 * \endcond
 * ================================================================================================
 */
GPU::~GPU()
{

}


/** ===============================================================================================
 * \name    cycle
 * 
 * \brief   Handling the GPU in a period of cycle
 * 
 * \endcond
 * ================================================================================================
 */
void
GPU::cycle()
{
    log_I("GPU Cycle", to_string(total_gpu_cycle));

    Check_Finish_Kernel();
    
    /* cycle() */
	for (auto& sm : mSMs) {
		sm.cycle();
	}

    /* gpu statistic */
    bool isBusy = false;
    for (auto& sm : mSMs) {
		isBusy |= sm.isRunning();
	}

    Runtime_Block_Scheduling();

}


/** ===============================================================================================
 * \name    Runtime_Block_Scheduling
 * 
 * \brief   Check whether the SM has resource to perform next kernel inside the commandQueue, and
 *          determine the proper block number for each kernel
 * 
 * \endcond
 * ================================================================================================
 */
void
GPU::Runtime_Block_Scheduling()
{  
    /* Iterate all kernels inside the command Queue */
    queue<Kernel*> remainingKernels;
    while(!commandQueue.empty())
    {
        Kernel* kernel = commandQueue.front();

        /* Sort the list by the corresponding resource */
#if (SM_SORT_TYPE == THEAD_NUM)
        mSMs.sort([](const SM& a, const SM& b){return a.getResourceInfo().remaining_threads > b.getResourceInfo().remaining_threads;});
#endif
        

        /* print sort result */
        for (auto& sm : mSMs) {
            cout << sm.smID << ", remain thread: " << sm.getResourceInfo().remaining_threads << endl;
        }

        /* bind kernel to SM by best-fit */
        SM& sm = mSMs.front();

        if (sm.bindKernel(kernel)) {
            runningKernels.push_back(kernel);
        } else {
            remainingKernels.push(kernel);
        }

        commandQueue.pop();
    }

    commandQueue = remainingKernels;
}


/** ===============================================================================================
 * \name    Check_Finish_Kernel
 * 
 * \brief   Check whether the kernel has been finished, move kernel from runningKernels to 
 *          finishedKernels.
 * 
 * \endcond
 * ================================================================================================
 */
void
GPU::Check_Finish_Kernel()
{  
    for (Kernel* kernel : runningKernels)
    {
        Block* b_info = kernel->record.block_info;
        if (b_info->finish)
        {
            kernel->record.end_cycle = total_gpu_cycle;

            finishedKernels.push_back(kernel);

            /* Recycle resource */
            kernel->record.running_sm->recycleResource(b_info);
        }
    }

    runningKernels.remove_if([](Kernel* k){return k->record.block_info->finish;});

    for(auto kernel: runningKernels)
    {
        cout << "running kernel id: " << kernel->kernelID << endl;
    }
}


/** ===============================================================================================
 * \name    launchKernel
 * 
 * \brief   Check whether the SM has resource to perform next kernel inside the commandQueue, and
 *          determine the proper block number for each kernel
 * 
 * \param   kernel  the kernel pointer going to launch into command queue
 * 
 * \return  true / false of add kernel
 * 
 * \endcond
 * ================================================================================================
 */
bool
GPU::launchKernel(Kernel* kernel)
{
    if (!kernel->requests.size()) return false;

    commandQueue.push(kernel);
    kernel->record.start_cycle = total_gpu_cycle;

    log_D("launchKernel", "kernel: " + to_string(kernel->kernelID) + " launch success");

    return true;
}