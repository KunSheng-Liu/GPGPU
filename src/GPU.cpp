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
    log_T("GPU", "Runtime_Block_Scheduling");

    bool mem_allocate;

    /* Iterate all kernels inside the command Queue */
    queue<Kernel*> remainingKernels;
    while(!commandQueue.empty())
    {
        Kernel* kernel = commandQueue.front();
        ASSERT(kernel, "Receive null kernel ptr");

        bool success = false;
        for (auto sm_id : *kernel->SM_List) success |= mSMs[sm_id].bindKernel(kernel);

        success ? runningKernels.push_back(kernel) : remainingKernels.push(kernel);
        mem_allocate = success;

        commandQueue.pop();
    }
    commandQueue = remainingKernels;

#if (LOG_LEVEL >= VERBOSE)
    for (auto kernel: runningKernels) log_V("running kernel id", to_string(kernel->kernelID));
#endif

    /* Allocate memory to running kernels */
    if (mem_allocate) Memory_Allocation();
}


/** ===============================================================================================
 * \name    Memory_Allocation
 * 
 * \brief   Allocate memory to the model follow the rule
 * 
 * \endcond
 * ================================================================================================
 */
void
GPU::Memory_Allocation()
{  
    if (command.MEM_MODE == MEM_ALLOCATION::None)
    {
        mGMMU.setCGroupSize(-1, DRAM_SPACE / PAGE_SIZE);
    }
    else if (command.MEM_MODE == MEM_ALLOCATION::Average)
    {
        map<int, int> memory_budget;
        for (auto kernel : runningKernels) memory_budget[kernel->modelID] += DRAM_SPACE / PAGE_SIZE / runningKernels.size();

        for (auto model_pair : memory_budget) mGMMU.setCGroupSize(model_pair.first, model_pair.second);
    }
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
    log_T("GPU", "Check_Finish_Kernel");
    for (Kernel* kernel : runningKernels)
    {
        if (kernel->requests.empty())
        {
            kernel->finish = true;
            for (int sm_id : *kernel->SM_List) kernel->finish &= mSMs[sm_id].checkKernelComplete(kernel);
            
            if (kernel->finish) 
            {
                kernel->endCycle = total_gpu_cycle;
                finishedKernels.push_back(kernel);
            }
        }
    }

    runningKernels.remove_if([](Kernel* k){return k->isFinish();});

    /* Allocate memory to running kernels */
    if (!finishedKernels.empty()) Memory_Allocation();
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

    log_V("launchKernel", "kernel: " + to_string(kernel->kernelID) + " launch success");

    return true;
}


/** ===============================================================================================
 * \name    terminateKernel
 * 
 * \brief   Erase the kernel from the GPU runningKernels queue, clear all used resource inlcudes in
 *          SM, Warp, GMMU
 * 
 * \param   model_id  the model id going to terminate
 * 
 * \return  true / false of terminate kernel
 * 
 * \endcond
 * ================================================================================================
 */
bool
GPU::terminateModel (int model_id)
{
    log_D("GPU", "terminateModel");
    /* Release GMMU */
    mGMMU.terminateModel(model_id);

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
    queue<Kernel*> remainingKernels;
    while(!commandQueue.empty())
    {
        Kernel* kernel = commandQueue.front();
        ASSERT(kernel, "Termiate kernel error");

        if (kernel->modelID == model_id) continue;

        commandQueue.pop();
    }
    commandQueue = remainingKernels;

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