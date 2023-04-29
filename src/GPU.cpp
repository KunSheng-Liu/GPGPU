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
    
    /* cycle() */
	for (auto& sm : mSMs) {
		sm.cycle();
	}

    /* check finish */
	for (auto& sm : mSMs) {
		if (sm.checkIsComplete()) {
			canIssueKernel = true;
		}
	}

    /* gpu statistic */
    bool isBusy = false;
    for (auto& sm : mSMs) {
		isBusy |= sm.isRunning();
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
    queue<Kernel*> waitingKernels = commandQueue;

    while(!waitingKernels.empty())
    {
        Kernel* kernel = waitingKernels.front();

        for (auto& sm : mSMs) {
            // cheack wheather the resource is enought for this kernel
        }

        waitingKernels.pop();
    }
}