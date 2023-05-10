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
GMMU::GMMU(GPU* gpu, MemoryController* mc) : mGPU(gpu), mMC(mc)
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
    log_D("GMMU", "Access_Processing");
    /* *******************************************************************
     * Receive the responce of Memory Controllor
     * *******************************************************************
     */
    if(!mMC->mc_to_gmmu_queue.empty()) log_D("MC", "Retrun " + to_string(mMC->mc_to_gmmu_queue.size()) + "access");
    gmmu_to_sm_queue.splice(gmmu_to_sm_queue.end(), mMC->mc_to_gmmu_queue);


    /* *******************************************************************
     * Collect the accesses from each SM
     * *******************************************************************
     */
    for (auto& sm : mGPU->mSMs) { 
        for (auto& warp : sm.second.mWarps)
        {
            sm_to_gmmu_queue.splice(sm_to_gmmu_queue.end(), warp.second.sm_to_gmmu_queue);
        }
	}
    if(!sm_to_gmmu_queue.empty()) log_D("GMMU", "Receive " + to_string(sm_to_gmmu_queue.size()) + "access");


    /* *******************************************************************
     * Handling the accesses
     * *******************************************************************
     */
    if(!sm_to_gmmu_queue.empty()) log_D("GMMU", "Handle " + to_string(sm_to_gmmu_queue.size()) + "access");
    for (auto access = sm_to_gmmu_queue.begin(); access != sm_to_gmmu_queue.end();)
    {
        Page* page;
        /* Hit */
        if (mCGroups[(*access)->model_id].second.lookup((*access)->page_id, page))
        {
            ((*access)->type == Read) ? page->info.read_counter++ : page->info.write_counter++;
            ++page->info.access_count;
            mMC->gmmu_to_mc_queue.splice(mMC->gmmu_to_mc_queue.end(), sm_to_gmmu_queue, access++);
        } 
        /* Miss */
        else {
            MSHRs.splice(MSHRs.end(), sm_to_gmmu_queue, access++);
        }
    }


    /* *******************************************************************
     * Return finished access to SM
     * *******************************************************************
     */
    if(!gmmu_to_sm_queue.empty()) log_D("GMMU", "Return " + to_string(gmmu_to_sm_queue.size()) + "access");
    while(!gmmu_to_sm_queue.empty())
    {
        MemoryAccess* access = gmmu_to_sm_queue.front();
        mGPU->mSMs[access->sm_id].mWarps[access->warp_id].gmmu_to_sm_queue.push_back(access);
        gmmu_to_sm_queue.pop_front();
    }
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
    log_D("GMMU", "Page_Fault_Handler");
    for (auto& access : MSHRs)
    {
        Page* page = mMC->access(access->page_id);
        if (page->info.location == SPACE_VRAM)
        {
            continue;
        }
        page->info.location = SPACE_VRAM;
        ++page->info.swap_count;

        Page* evicted_page = mCGroups[access->model_id].second.insert(access->page_id, page);
        if (evicted_page)
        {
            evicted_page->info.location = SPACE_DRAM;
            ++evicted_page->info.swap_count;
        }
    }
    sm_to_gmmu_queue.splice(sm_to_gmmu_queue.end(), MSHRs);
}


/** ===============================================================================================
 * \name    setCGroupSize
 * 
 * \brief   Set the CGroup Size to the sepcific model
 * 
 * \param   model_id    the index of model
 * \param   capacity    the capacity of the model's cgroup
 * 
 * \endcond
 * ================================================================================================
 */
void
GMMU::setCGroupSize (int model_id, unsigned capacity)
{
    mCGroups[model_id].second.resize(capacity);
    mCGroups[model_id].first = capacity;

    if (mCGroups.find(model_id) != mCGroups.end())
    {
        cout << "setCGroupSize: [" << model_id << ", " << mCGroups[model_id].first << "]" << endl;
    }
    
}


