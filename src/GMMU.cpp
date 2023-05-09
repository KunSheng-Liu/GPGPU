/**
 * \name    GMMU.cpp
 * 
 * \brief   Declare the structure of GMMU
 * 
 * \date    May 9, 2023
 */

#include "include/GPU.hpp"
#include "include/GMMU.hpp"

/** ===============================================================================================
 * \name    GMMU
 * 
 * \brief   The class of the GMMU
 * 
 * \endcond
 * ================================================================================================
 */
GMMU::GMMU(GPU* gpu) : mGPU(gpu)
{

}


/** ===============================================================================================
 * \name    GMMU
 * 
 * \brief   Destruct GMMU
 * 
 * \endcond
 * ================================================================================================
 */
GMMU::~GMMU()
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
GMMU::cycle()
{
    log_I("GMMU Cycle", to_string(total_gpu_cycle));
    
    Page_Fault_Handler();

    Access_Processing();

    /* Return finished access to SM */
    if(!gmmu_to_sm_queue.empty()) log_D("GMMU", "Return " + to_string(gmmu_to_sm_queue.size()) + "access");
    while(!gmmu_to_sm_queue.empty())
    {
        MemoryAccess* access = gmmu_to_sm_queue.front();
        mGPU->mSMs[access->sm_id].mWarps[access->warp_id].gmmu_to_sm_queue.push_back(move(access));
        gmmu_to_sm_queue.pop();
    }

    /* Collect the accesses from each SM */
    for (auto& sm : mGPU->mSMs) { 
        for (auto& warp : sm.second.mWarps)
        {
            while(!warp.second.sm_to_gmmu_queue.empty())
            {
                sm_to_gmmu_queue.push(move(warp.second.sm_to_gmmu_queue.front()));
                warp.second.sm_to_gmmu_queue.pop_front();
            }
        }
	}
    if(!sm_to_gmmu_queue.empty()) log_D("GMMU", "Receive " + to_string(gmmu_to_sm_queue.size()) + "access");

    /* Handling the accesses */
    if(!sm_to_gmmu_queue.empty()) log_D("GMMU", "Handle " + to_string(sm_to_gmmu_queue.size()) + "access");
    while(!sm_to_gmmu_queue.empty())
    {
        gmmu_to_sm_queue.push(move(sm_to_gmmu_queue.front()));
        sm_to_gmmu_queue.pop();
    }
}


/** ===============================================================================================
 * \name    Access_Processing
 * 
 * \brief   Process the access inside the sm_to_gmmu_queue
 * 
 * \endcond
 * ================================================================================================
 */
void
GMMU::Access_Processing()
{

}


/** ===============================================================================================
 * \name    Page_Fault_Handler
 * 
 * \brief   Handling the page fault and the penalty
 * 
 * \endcond
 * ================================================================================================
 */
void
GMMU::Page_Fault_Handler()
{

}


