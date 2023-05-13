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
    log_T("GMMU", "Access_Processing");
    /* *******************************************************************
     * Receive the responce of Memory Controllor
     * *******************************************************************
     */
    if(!mMC->mc_to_gmmu_queue.empty()) 
    {
        log_T("MC", "Retrun " + to_string(mMC->mc_to_gmmu_queue.size()) + " access");
        gmmu_to_sm_queue.splice(gmmu_to_sm_queue.end(), mMC->mc_to_gmmu_queue);
    }
    

    /* *******************************************************************
     * Return finished access to SM
     * *******************************************************************
     */
    if(!gmmu_to_sm_queue.empty()) log_T("GMMU", "Return " + to_string(gmmu_to_sm_queue.size()) + " access");
    while(!gmmu_to_sm_queue.empty())
    {
        MemoryAccess* access = gmmu_to_sm_queue.front();
        mGPU->mSMs[access->sm_id].mWarps[access->warp_id].gmmu_to_sm_queue.push_back(access);
        gmmu_to_sm_queue.pop_front();
    }


    /* *******************************************************************
     * Collect the accesses from each SM
     * *******************************************************************
     */
    for (auto& sm : mGPU->mSMs) { 
        for (auto& warp : sm.second.mWarps)
        {
            if (!warp.second.sm_to_gmmu_queue.empty()) sm_to_gmmu_queue.splice(sm_to_gmmu_queue.end(), warp.second.sm_to_gmmu_queue);
        }
	}
    if(!sm_to_gmmu_queue.empty()) log_T("GMMU", "Receive " + to_string(sm_to_gmmu_queue.size()) + " access");


    /* *******************************************************************
     * Handling the accesses
     * *******************************************************************
     */
    if(!sm_to_gmmu_queue.empty()) log_T("GMMU", "Handle " + to_string(sm_to_gmmu_queue.size()) + " access");
    while(!sm_to_gmmu_queue.empty())
    {
        MemoryAccess* access = sm_to_gmmu_queue.front();
        auto model_id = access->model_id;

        /* Check whether the page of current access is in the memory */
        bool hit = true;
        for (auto page_id : access->pageIDs) hit &= mCGroups[model_id].second.lookup(page_id);

        /* Classify the access into correspond handling queue */
        hit ? mMC->gmmu_to_mc_queue.push_back(access) : MSHRs.push_back(access);

        sm_to_gmmu_queue.pop_front();
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
    log_T("GMMU", "Page_Fault_Handler");

    /* *******************************************************************
     * Waiting for communication to the CPU and migration overhead
     * *******************************************************************
     */
    if (wait_cycle > 0)
    {
        log_V("Page_Fault_Handler cycle", to_string(wait_cycle--));
        return;
    } 
    else {
        /* *******************************************************************
         * Perform page movement after finish delay of communication and 
         * migration
         * *******************************************************************
         */
        if (!page_fault_process_queue.empty())
        {
            for (auto& fault_pair : page_fault_process_queue)
            {
                auto model_id = fault_pair.first;

                ASSERT(fault_pair.second.size() <= mCGroups[model_id].first, "Allocated memory is less than the model needed");

                for (auto& page_id : fault_pair.second)
                {
                    Page* page;
                    /* Migration from DRAM to VRAM */
                    ASSERT(mMC->refer(page_id), "page ptr doesn't exist");
                    page->record.location = SPACE_VRAM;
                    page->record.swap_count++;
                    page = mCGroups[model_id].second.insert(page_id, page);

                    /* Eviction happen */
                    if (page)
                    {
                        page->record.location = SPACE_DRAM;
                        page->record.swap_count++;
                    }
                }
            }
            page_fault_process_queue.clear();
            /* *******************************************************************
            * Handle return
            * *******************************************************************
            */
            sm_to_gmmu_queue.splice(sm_to_gmmu_queue.end(), page_fault_finish_queue);
        }
    
        /* *******************************************************************
         * Launch the access inside the MSHRs to handling queue, not remove the
         * access from the MSHRs until processing over.
         * *******************************************************************
         */
        if (!MSHRs.empty())
        {
            for (auto access : MSHRs)
            {
                for (auto page_id : access->pageIDs)
                {
                    page_fault_process_queue[access->model_id].insert(page_id);
                }
                
            }
            page_fault_finish_queue.splice(page_fault_finish_queue.end(), MSHRs);
            // wait_cycle = PAGE_FAULT_COMMUNICATION_CYCLE + page_fault_process_queue.size() * PAGE_FAULT_MIGRATION_UNIT_CYCLE;
            wait_cycle = 1;
        }
    }
    
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

#if (LOG_LEVEL >= VERBOSE)
    std::cout << "setCGroupSize: [" << model_id << ", " << mCGroups[model_id].first << "]" << std::endl;
#endif
    
}


/** ===============================================================================================
 * \name    freeCGroup
 * 
 * \brief   Free up the CGroup of specific model
 * 
 * \param   model_id    the index of model
 * 
 * \endcond
 * ================================================================================================
 */
void
GMMU::freeCGroup (int model_id)
{
    auto it = mCGroups.find(model_id);
    if (it != mCGroups.end()) mCGroups.erase(it);
}