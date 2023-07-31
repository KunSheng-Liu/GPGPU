/**
 * \name    Scheduler_Baseline.cpp
 * 
 * \brief   Implement the basic function used in CPU.hpp.
 * 
 * \date    Jun 22, 2023
 */
#include "include/Scheduler.hpp"

/** ===============================================================================================
 * \name    Scheduler_Baseline
 * 
 * \brief   ...
 * 
 * \param   cpu     the pointer of CPU
 * 
 * \endcond
 * ================================================================================================
 */ 
Scheduler_Baseline::Scheduler_Baseline (CPU* cpu) : Scheduler(cpu)
{

}


/** ===============================================================================================
 * \name    Sched
 * 
 * \brief   perform schedule
 * 
 * \endcond
 * ================================================================================================
 */
void 
Scheduler_Baseline::Sched ()
{
    Inference_Admission();

    Memory_Allocator();
    
    Inference_Launcher();
}


/** ===============================================================================================
 * \name    Inference_Admission
 * 
 * \brief   Assign all SM to every applications
 * 
 * \note    * This defualt scheduler didn't block SM to any application, which leads the kernel contension
 *          the SM resource in GPU core.
 * 
 * \endcond
 * ================================================================================================
 */
bool 
Scheduler_Baseline::Inference_Admission ()
{
    unordered_set<int> available_sm;
    for (int i = 0; i < system_resource.SM_NUM; i++) available_sm.insert(i);

    for (auto app : mCPU->mAPPs) app->SM_budget = available_sm;

    for (auto app : mCPU->mAPPs) app->runningModels.splice(app->runningModels.end(), app->waitingModels);

    return true;
}


/** ===============================================================================================
 * \name    Memory_Allocator
 * 
 * \brief   Share memeory space to all applications
 * 
 * \endcond
 * ================================================================================================
 */
bool 
Scheduler_Baseline::Memory_Allocator ()
{
    mCPU->mGPU->getGMMU()->setCGroupType (false);
    mCPU->mGPU->getGMMU()->setCGroupSize(-1, system_resource.VRAM_SPACE / PAGE_SIZE);
    return true;
}


/** ===============================================================================================
 * \name    Inference_Launcher
 * 
 * \brief   Launch all ready kerenls to GPU command queue with max batch size
 * 
 * \endcond
 * ================================================================================================
 */
bool
Scheduler_Baseline::Inference_Launcher ()
{
    for (auto app : mCPU->mAPPs)
    {
        /* *******************************************************************
         * Collect the ready kernels
         * *******************************************************************
         */
        list<Kernel*> readyKernels;
        for (auto model : app->runningModels)
        {
            for (auto kernel : model->findReadyKernels()) readyKernels.push_back(kernel);
        }
        if (readyKernels.empty()) continue;
        
        /* print ready list */
#if (LOG_LEVEL >= VERBOSE)
        std::cout << "App " << app.appID << ": Ready kernel list: ";
        for (auto kernel : readyKernels)
        {
            std::cout << kernel->kernelID << ", ";
        }
        std::cout << std::endl;
#endif
        /* Collect the kernel with smallest layer ID */
        vector<pair<Kernel*, int>> sync_kernels;
        for (auto k : readyKernels)
        {
            if (k->srcLayer->layerID == readyKernels.front()->srcLayer->layerID) 
            {
                sync_kernels.push_back(make_pair(k, 1));
            }
        }
        
        /* Transform into kernelGroup */
        Kernel* kernel  = new KernelGroup(sync_kernels);
        kernel->SM_List = new unordered_set<int> (app->SM_budget);
        
        /* *******************************************************************
         * Launch kernel to inference
         * *******************************************************************
         */
        kernelLauncher(move(kernel));
    }
    return true;
}