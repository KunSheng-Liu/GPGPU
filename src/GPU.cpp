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
GPU::GPU(MemoryController* mc) : mMC(mc), mGMMU(GMMU())
{
    /* Create SMs */
    for (int i = 0; i < GPU_SM_NUM; i++)
    {
        mSMs.insert(make_pair(i, SM()));
        mSMs[i].setGMMU(&mGMMU);
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

    
    /* cycle() */
	for (auto& sm : mSMs) {
		sm.second.cycle();
	}

    /* Check finish() */
	for (auto& sm : mSMs) {
		sm.second.checkFinish();
	}

    Check_Finish_Kernel();

    Runtime_Block_Scheduling();

    /* gpu statistic */
    bool isBusy = false;
    for (auto& sm : mSMs) {
		isBusy |= sm.second.isRunning();
	}

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
        commandQueue.pop();

        bool success = false;
        for (auto sm_id : kernel->record->SM_List)
        {
            cout << "Launch kernel: " << kernel->kernelID << " to SM: " << sm_id << endl;
            success |= mSMs[sm_id].bindKernel(kernel);
        }

        if (success) {
            runningKernels.push_back(kernel);
        } else {
            remainingKernels.push(kernel);
        }

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
        if (kernel->requests.empty())
        {
            kernel->finish = true;
            for (int sm_id : kernel->record->SM_List)
            {
                kernel->finish &= mSMs[sm_id].checkIsComplete(kernel);
            }
        }

        if (kernel->isFinish()) 
        {
            kernel->end_cycle = total_gpu_cycle;
            finishedKernels.push_back(kernel);
        }
    }

    runningKernels.remove_if([](Kernel* k){return k->isFinish();});

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

    log_D("launchKernel", "kernel: " + to_string(kernel->kernelID) + " launch success");

    return true;
}