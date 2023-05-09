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

    /* Return finished access to SM */
    if(!gmmu_to_sm_access.empty()) log_D("GMMU", "Return " + to_string(gmmu_to_sm_access.size()) + "access");
    while(!gmmu_to_sm_access.empty())
    {
        MemoryAccess* access = gmmu_to_sm_access.front();
        mGPU->mSMs[access->sm_id].gmmu_to_sm_access.push_back(move(access));
        gmmu_to_sm_access.pop();
    }

    /* Handling the accesses */
    if(!sm_to_gmmu_access.empty()) log_D("GMMU", "Handle " + to_string(sm_to_gmmu_access.size()) + "access");
    while(!sm_to_gmmu_access.empty())
    {
        gmmu_to_sm_access.push(move(sm_to_gmmu_access.front()));
        sm_to_gmmu_access.pop();
    }

    /* Collect the accesses from each SM */
    for (auto& sm : mGPU->mSMs) { 

        while(!sm.second.sm_to_gmmu_access.empty())
        {
            sm_to_gmmu_access.push(move(sm.second.sm_to_gmmu_access.front()));
            sm.second.sm_to_gmmu_access.pop_front();
        }
	}
    if(!sm_to_gmmu_access.empty()) log_D("GMMU", "Receive " + to_string(gmmu_to_sm_access.size()) + "access");
}
