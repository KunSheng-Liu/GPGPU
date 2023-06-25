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
#if (LOG_LEVEL >= TRACE)
    log_T("GMMU", "Access_Processing");
#endif
    /* *******************************************************************
     * Receive the responce of Memory Controllor
     * *******************************************************************
     */
#if (LOG_LEVEL >= TRACE)
    log_T("MC", "Retrun " + to_string(mMC->mc_to_gmmu_queue.size()) + " access");
#endif
    if(!mMC->mc_to_gmmu_queue.empty()) 
    {
        gmmu_to_warps_queue.splice(gmmu_to_warps_queue.end(), mMC->mc_to_gmmu_queue);
    }
    

    /* *******************************************************************
     * Return finished access to SM
     * *******************************************************************
     */
#if (LOG_LEVEL >= TRACE)
    log_T("GMMU", "Return " + to_string(gmmu_to_warps_queue.size()) + " access");
#endif
    while(!gmmu_to_warps_queue.empty())
    {
        auto access = gmmu_to_warps_queue.front();
        mGPU->mSMs[access->sm_id].mWarps[access->warp_id].gmmu_to_warp_queue.push_back(access);
        gmmu_to_warps_queue.pop_front();
    }


    /* *******************************************************************
     * Collect the accesses
     * *******************************************************************
     */
    while(1)
    {
        bool empty = true;
        for (int j = 0; j < GPU_MAX_WARP_PER_SM; j++)
        {
            for (int i = 0; i < system_resource.SM_NUM; i++)
            {
                if (!mGPU->mSMs[i].mWarps[j].warp_to_gmmu_queue.empty())
                {
                    warps_to_gmmu_queue.push_back(mGPU->mSMs[i].mWarps[j].warp_to_gmmu_queue.front());
                    mGPU->mSMs[i].mWarps[j].warp_to_gmmu_queue.pop_front();
                    empty = false;
                }
            }
        }
        if (empty) break;
    }


    /* *******************************************************************
     * Handling the accesses
     * *******************************************************************
     */
#if (LOG_LEVEL >= TRACE)
    log_T("GMMU", "Handle " + to_string(warps_to_gmmu_queue.size()) + " access");
#endif
    while(!warps_to_gmmu_queue.empty())
    {
        auto access = warps_to_gmmu_queue.front();
        auto TLB = getCGroup(access->app_id);

        /* Check whether the page of current access is in the memory */
        bool hit = true;
        Page* dummy_page;
        for (auto page_id : access->pageIDs) hit &= TLB->lookup(page_id, dummy_page);

        /* Classify the access into correspond handling queue */
        hit ? mMC->gmmu_to_mc_queue.push_back(access) : MSHRs.push_back(access);

        warps_to_gmmu_queue.pop_front();
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
#if (LOG_LEVEL >= TRACE)
    log_T("GMMU", "Page_Fault_Handler");
#endif
    /* *******************************************************************
     * Waiting for communication to the CPU and migration overhead
     * *******************************************************************
     */
    if (wait_cycle > 0)
    {
        wait_cycle--;
#if (LOG_LEVEL >= VERBOSE)
        log_V("Page_Fault_Handler cycle", to_string(wait_cycle));
#endif
    }

    /* *******************************************************************
     * Perform page movement after finish delay of communication and 
     * migration
     * *******************************************************************
     */
    else if (!page_fault_process_queue.empty())
    {
        auto access_pair = page_fault_process_queue.back();

        /* Migration from DRAM to VRAM */
        unsigned long long page_id = access_pair.first;
        Page* page = mMC->refer(page_id);
        page->location = SPACE_VRAM;
        page->record.swap_count++;

        for (auto access : access_pair.second)
        {
            Page* evict_page = getCGroup(access->app_id)->insert(page_id, page);

            /* Eviction happen */
            if (evict_page)
            {
                evict_page->location = SPACE_DRAM;
                evict_page->record.swap_count++;
                wait_cycle = PAGE_FAULT_MIGRATION_UNIT_CYCLE;
            }

            if (--access_count[access] == 0)
            {
                warps_to_gmmu_queue.push_back(access);
                access_count.erase(access_count.find(access));
            }
        }

        page_fault_process_queue.pop_back();

#if (ENABLE_PAGE_FAULT_PENALTY)
        wait_cycle += PAGE_FAULT_MIGRATION_UNIT_CYCLE;
#else            
        wait_cycle = 1;
#endif
    }
    
    /* *******************************************************************
     * Launch the access inside the MSHRs to handling queue, not remove the
     * access from the MSHRs until processing over.
     * *******************************************************************
     */
    else if (!MSHRs.empty())
    {
        map<int, unordered_set<int>> access_record;
        unordered_map<unsigned long long, list<MemoryAccess*>> page_fault_record;
        /* *******************************************************************
         * Find the demanded pages
         * *******************************************************************
         */
        list<MemoryAccess*> remaining_MSHRs = {};
        for (auto access : MSHRs)
        {
            list<unsigned long long> page_list = {};
            for (auto page_id : access->pageIDs) if (!getCGroup(access->app_id)->lookup(page_id)) page_list.push_back(page_id);
            if (page_list.empty())
            {
                warps_to_gmmu_queue.push_back(access);
                continue;
            }

            int new_page = 0;
            for (auto page_id : page_list) if (!access_record[access->app_id].count(page_id)) new_page++;
            if (access_record[access->app_id].size() + new_page > getCGroup(access->app_id)->size() || page_fault_record.size() + new_page > PCIE_ACCESS_BOUND)
            {
                remaining_MSHRs.push_back(access);
                continue;
            }

            /* add the page access into queue */
            access_record[access->app_id].insert(page_list.begin(), page_list.end());
            for (auto page_id : page_list) page_fault_record[page_id].push_back(access);
            access_count[access] += page_list.size();
        }

        MSHRs = remaining_MSHRs;
        page_fault_process_queue = list<pair<unsigned long long, list<MemoryAccess*>>>(page_fault_record.begin(), page_fault_record.end());
        
#if (PAGE_PREFETCH)
        /* *******************************************************************
         * Prefetch smaller gap first
         * *******************************************************************
         */
        if (page_count < PCIE_ACCESS_BOUND)
        {
            list<pair<int, size_t>> cgroup_pairs = {}; // first: key, second: fillup_gap
            for (auto CGroup : mCGroups) cgroup_pairs.emplace_back(make_pair(CGroup.first, CGroup.second.size() - CGroup.second.usage()));

            cgroup_pairs.sort([](const auto& a, const auto& b){ return a.second > b.second; });
            
            for (auto pair : cgroup_pairs)
            {
                int prefetch_limit = min((int)(PCIE_ACCESS_BOUND - page_count), (int)pair.second);
                
                unordered_set<unsigned long long> prefetch_list = {};
                for (auto page_id : page_fault_process_queue[pair.first])
                {
                    auto page = mMC->mPages[page_id].nextPage;

                    while (page) 
                    {
                        if (prefetch_list.size() == prefetch_limit) break;

                        if (!mCGroups[pair.first].lookup(page->pageIndex)) prefetch_list.insert(page->pageIndex);
                        page = page->nextPage;
                    }
                }

                page_fault_process_queue[pair.first].insert(prefetch_list.begin(), prefetch_list.end());
                page_count += prefetch_list.size();

                if (page_count == PCIE_ACCESS_BOUND) break;
            }
        }
#endif

#if (ENABLE_PAGE_FAULT_PENALTY)
        wait_cycle = PAGE_FAULT_COMMUNICATION_CYCLE + PAGE_FAULT_MIGRATION_UNIT_CYCLE;
#else            
        wait_cycle = 1;
#endif

        log ("Demanded page number", to_string(page_fault_process_queue.size()), Color::Cyan);
#if (PRINT_DEMAND_PAGE_RECORD)
        ofstream file (LOG_OUT_PATH + program_name + ".txt", std::ios::app);
            file << "Demanded page number: " << page_fault_process_queue.size() << std::endl;
        file.close();
#endif
            
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
    warps_to_gmmu_queue.remove_if([model_id](MemoryAccess* access){return access->model_id == model_id;});
    gmmu_to_warps_queue.remove_if([model_id](MemoryAccess* access){return access->model_id == model_id;});

    for (auto page_pair : page_fault_process_queue)
    {
        page_pair.second.remove_if([model_id](MemoryAccess* access){return access->model_id == model_id;});
    }
    page_fault_process_queue.remove_if([](auto& pair){return pair.second.empty();});
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
GMMU::setCGroupSize (int app_id, unsigned long long capacity)
{
    mCGroups[app_id].resize(capacity);

    log("setCGroupSize", "[" + to_string(app_id) + ", " + to_string(mCGroups[app_id].size()) + "]", Color::Cyan);
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
        int release_count = (*it).second.release( check );
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
LRU_TLB<unsigned long long, Page*>*
GMMU::getCGroup (int app_id)
{
    return &mCGroups[(command.MEM_MODE == MEM_ALLOCATION::None) ? -1 : app_id];
}