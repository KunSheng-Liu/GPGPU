/**
 * \name    Memory_Allocator.cpp
 * 
 * \brief   Implement the memory allocation API, allocate GPU unified memory (DRAM + VRAM) space 
 *          to every application.
 * 
 * \date    Jun 22, 2023
 */
#include "include/Scheduler.hpp"


/** ===============================================================================================
 * \name    None
 * 
 * \brief   All application share same memory space
 * 
 * \endcond
 * ================================================================================================
 */
bool 
Memory_Allocator_API::None (CPU* mCPU)
{  
    mCPU->mGPU->getGMMU()->setCGroupSize(-1, system_resource.VRAM_SPACE / PAGE_SIZE);
    return true;
}


/** ===============================================================================================
 * \name    Average
 * 
 * \brief   Average allocates memory space to each application
 * 
 * \endcond
 * ================================================================================================
 */
bool 
Memory_Allocator_API::Average (CPU* mCPU)
{  
    map<int, int> memory_budget;

    int size = system_resource.VRAM_SPACE / PAGE_SIZE;

    for (auto app : mCPU->mAPPs) memory_budget[app->appID] += floor(size / mCPU->mAPPs.size());

    for (int i = 0; i < size % mCPU->mAPPs.size(); i++) memory_budget[mCPU->mAPPs[i]->appID]++;
    
    for (auto app : mCPU->mAPPs) mCPU->mGPU->getGMMU()->setCGroupSize(app->appID, memory_budget[app->appID]);
    
    return true;
}