/**
 * \name    GPU.cpp
 * 
 * \brief   Implement the CPU.
 * 
 * \date    APR 30, 2023
 */

#include "include/GPU.hpp"
#include "include/GMMU.hpp"

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
GPU::GPU(MemoryController* mc) : mMC(mc), mGMMU(GMMU(this, mc))
{
    /* Create SMs */
    for (int i = 0; i < system_resource.SM_NUM; i++)
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
    ASSERT(commandQueue.empty(), "Error Destruct");
    ASSERT(runningKernels.empty(), "Error Destruct");
    ASSERT(finishedKernels.empty(), "Error Destruct");
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

    /* Check finish() */
	for (auto& sm : mSMs) {
		sm.second.checkBlockFinish();
	}

    Check_Finish_Kernel();

    Runtime_Block_Scheduling();

    /* cycle() */
	for (auto& sm : mSMs) {
		sm.second.cycle();
	}

    /* gpu statistic */
    statistic();
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
#if (LOG_LEVEL >= TRACE)
    log_T("GPU", "Runtime_Block_Scheduling");
#endif
    /* Iterate all kernels inside the command Queue */
    list<Kernel*>  remainingQueue = {};
    for (auto kernel : commandQueue)
    {
        ASSERT(kernel, "Receive null kernel ptr");

        bool sm_ready = true;
        for (auto sm_id : *kernel->SM_List) sm_ready &= mSMs[sm_id].isIdel();

        if (sm_ready)
        {
            int division_count = kernel->SM_List->size() * GPU_MAX_WARP_PER_SM / GPU_MAX_WARP_PER_BLOCK;

            int num_of_request = ceil((float)kernel->requests.size() / division_count);
            for (auto sm_id : *kernel->SM_List) mSMs[sm_id].bindKernel(kernel, num_of_request);
            
            ASSERT(kernel->requests.empty(), "error");
            
            runningKernels.push_back(kernel);
        }
        else remainingQueue.push_back(kernel);
    }
    
    commandQueue = remainingQueue;

#if (LOG_LEVEL >= VERBOSE)
    for (auto kernel: runningKernels) log_V("running kernel id", to_string(kernel->kernelID));
#endif
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
#if (LOG_LEVEL >= TRACE)
    log_T("GPU", "Check_Finish_Kernel");
#endif
    for (auto kernel : runningKernels)
    {
        bool finish = true;
        for (auto sm_id : *kernel->SM_List) finish &= mSMs[sm_id].checkKernelComplete(kernel);
        
        if (finish) 
        {
            kernel->endCycle = total_gpu_cycle;
            finishedKernels.push_back(kernel);
        }
    }

    runningKernels.remove_if([this](Kernel* k){ return find(this->finishedKernels.begin(), this->finishedKernels.end(), k) != this->finishedKernels.end(); });
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

    commandQueue.push_back(kernel);

    log_V("launchKernel", "kernel: " + to_string(kernel->kernelID) + " launch success");

    return true;
}


/** ===============================================================================================
 * \name    terminateKernel
 * 
 * \brief   Erase the kernel from the GPU runningKernels queue, clear all used resource inlcudes in
 *          SM, Warp, GMMU
 * 
 * \param   app_id    the application id of the model
 * 
 * \param   model_id  the model id going to terminate
 * 
 * \return  true / false of terminate kernel
 * 
 * \endcond
 * ================================================================================================
 */
bool
GPU::terminateModel (int app_id, int model_id)
{
    log_D("GPU", "terminateModel");
    /* Release GMMU */
    mGMMU.terminateModel(app_id, model_id);

    /* Release SM */
    for (auto kernel : runningKernels)
    {
        if (kernel->modelID == model_id)
        {
            for (auto& sm : mSMs) sm.second.terminateKernel(kernel);
            kernel->running = false;
        }
    }
    runningKernels.remove_if([](Kernel* k){return !k->running;});

    /* Release commandQueue */
    commandQueue.remove_if([model_id](Kernel* k){return !k->modelID == model_id;});

    return true;
}


/** ===============================================================================================
 * \name    statistic
 * 
 * \brief   Record the SM runtime information
 * 
 * \endcond
 * ================================================================================================
 */
void
GPU::statistic()
{
    // bool isBusy = false;
    // for (auto& sm : mSMs) {
	// 	isBusy |= sm.second.isRunning();
	// }
}


/** ===============================================================================================
 * \name    getIdleSMs
 * 
 * \brief   Return the available SM index list
 * 
 * \return  list of sm index
 * 
 * \endcond
 * ================================================================================================
 */
unordered_set<int>
GPU::getIdleSMs()
{
    unordered_set<int> available_list = {};
    
    for (auto& sm : mSMs) if (sm.second.isIdel()) available_list.insert(sm.first);    

    return available_list;
}