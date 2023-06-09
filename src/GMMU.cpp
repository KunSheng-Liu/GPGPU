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
    log_T("MC", "Retrun " + to_string(mMC->mc_to_gmmu_queue.size()) + " access");
    if(!mMC->mc_to_gmmu_queue.empty()) 
    {
        gmmu_to_sm_queue.splice(gmmu_to_sm_queue.end(), mMC->mc_to_gmmu_queue);
    }
    

    /* *******************************************************************
     * Return finished access to SM
     * *******************************************************************
     */
    log_T("GMMU", "Return " + to_string(gmmu_to_sm_queue.size()) + " access");
    while(!gmmu_to_sm_queue.empty())
    {
        auto access = gmmu_to_sm_queue.front();
        mGPU->mSMs[access->sm_id].mWarps[access->warp_id].gmmu_to_sm_queue.push_back(access);
        gmmu_to_sm_queue.pop_front();
    }


    /* *******************************************************************
     * Handling the accesses
     * *******************************************************************
     */
    log_T("GMMU", "Handle " + to_string(sm_to_gmmu_queue.size()) + " access");
    while(!sm_to_gmmu_queue.empty())
    {
        auto access = sm_to_gmmu_queue.front();
        auto TLB = getCGroup(access->app_id);

        /* Check whether the page of current access is in the memory */
        bool hit = true;
        for (auto page_id : access->pageIDs) hit &= TLB->second.lookup(page_id);

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
    } 
    else {
        /* *******************************************************************
         * Perform page movement after finish delay of communication and 
         * migration
         * *******************************************************************
         */
        if (!page_fault_process_queue.empty())
        {
            for (auto fault_pair : page_fault_process_queue)
            {
                ASSERT(fault_pair.second.size() <= getCGroup(fault_pair.first)->first, "Allocated memory is less than the model needed");

                for (auto page_id : fault_pair.second)
                {
                    /* Migration from DRAM to VRAM */
                    Page* page = mMC->refer(page_id);
                    ASSERT(page, "page ptr doesn't exist");
                    page->location = SPACE_VRAM;
                    page->record.swap_count++;
                    page = getCGroup(fault_pair.first)->second.insert(page_id, page);

                    /* Eviction happen */
                    if (page)
                    {
                        page->location = SPACE_DRAM;
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
                    page_fault_process_queue[access->app_id].insert(page_id);
                }
            }
            page_fault_finish_queue.splice(page_fault_finish_queue.end(), MSHRs);
            wait_cycle = PAGE_FAULT_COMMUNICATION_CYCLE + page_fault_process_queue.size() * PAGE_FAULT_MIGRATION_UNIT_CYCLE;
            // wait_cycle = 1;
        }
    }
    
}


/** ===============================================================================================
 * \name    terminateModel
 * 
 * \brief   Erase all access from queues if the model is terminate
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
GMMU::terminateModel(int app_id, int model_id)
{
    sm_to_gmmu_queue.remove_if([model_id](MemoryAccess* access){return access->model_id == model_id;});
    gmmu_to_sm_queue.remove_if([model_id](MemoryAccess* access){return access->model_id == model_id;});

    page_fault_finish_queue.remove_if([model_id](MemoryAccess* access){return access->model_id == model_id;});
    page_fault_process_queue.erase(model_id);
    if (page_fault_process_queue.empty()) wait_cycle = 0;

    mMC->mc_to_gmmu_queue.remove_if([model_id](MemoryAccess* access){return access->model_id == model_id;});
    mMC->gmmu_to_mc_queue.remove_if([model_id](MemoryAccess* access){return access->model_id == model_id;});

    freeCGroup(app_id);

    return true;
}


/** ===============================================================================================
 * \name    setCGroupSize
 * 
 * \brief   Set the CGroup Size to the sepcific model
 * 
 * \param   app_id      the index of application
 * \param   capacity    the capacity of the model's cgroup
 * 
 * \endcond
 * ================================================================================================
 */
void
GMMU::setCGroupSize (int app_id, unsigned capacity)
{
    mCGroups[app_id].second.resize(capacity);
    mCGroups[app_id].first = capacity;

#if (LOG_LEVEL >= VERBOSE)
    std::cout << "setCGroupSize: [" << model_id << ", " << mCGroups[model_id].first << "]" << std::endl;
#endif
    
}


/** ===============================================================================================
 * \name    freeCGroup
 * 
 * \brief   Free up the CGroup of specific model
 * 
 * \param   app_id    the index of application
 * 
 * \endcond
 * ================================================================================================
 */
inline bool check (Page* const& page) { return page->location == SPACE_DRAM; }
/*
 * ================================================================================================
 */
void
GMMU::freeCGroup (int app_id)
{
    auto it = mCGroups.find((command.MEM_MODE == MEM_ALLOCATION::None) ? -1 : app_id);
    if (it != mCGroups.end()) 
    {
        int release_count = (*it).second.second.release( check );
        log_V("freeCGroup", "release " + to_string(release_count) + " pages from the CGroup " + to_string((*it).first));
    }
}


/** ===============================================================================================
 * \name    getCGroup
 * 
 * \brief   Get the corresponding CGroup pointer
 * 
 * \param   app_id    the index of application
 * 
 * \endcond
 * ================================================================================================
 */
pair<int, LRU_TLB<unsigned long long, Page*>>*
GMMU::getCGroup (int app_id)
{
    return &mCGroups[(command.MEM_MODE == MEM_ALLOCATION::None) ? -1 : app_id];
}