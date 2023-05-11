/**
 * \name    SM.cpp
 * 
 * \brief   Implement the SM and it's components.
 * 
 * \date    May 7, 2023
 */

#include "include/SM.hpp"

/* ************************************************************************************************
 * Global Variable
 * ************************************************************************************************
 */
int SM::SMCount = 0;
int Block::blockCount = 0;

/** ===============================================================================================
 * \name    SM
 * 
 * \brief   The class of the SM
 * 
 * \param   mc      the pointer of memory controller
 * 
 * \endcond
 * ================================================================================================
 */
SM::SM() : smID(SMCount++)
{
    /* Create Warps */
    for (int i = 0; i < GPU_MAX_WARP_PER_SM; i++)
    {
        mWarps.insert(make_pair(i, Warp(i)));
    }
}


/** ===============================================================================================
 * \name    SM
 * 
 * \brief   Destruct SM
 * 
 * \endcond
 * ================================================================================================
 */
SM::~SM()
{
    ASSERT(runningBlocks.empty(), "Error Destruct");
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
SM::cycle()
{    
    log_I("SM " + to_string(smID) + " Cycle", to_string(total_gpu_cycle));

    /* SM statistic */
    if (isRunning()) {
        info.exec_cycle++;

        if (isComputing()) {
            info.computing_cycle++;
        }
        else {
            info.wait_cycle++;
        }
    }
    else {
        info.idle_cycle++;
    }

    /* Computing */
    for (auto& block : runningBlocks)
    {
        log_V("SM", to_string(smID) + " Execute block: " + to_string(block->block_id));

        int access_count = 0;
        for (auto& warp : block->warps)
        {
            /* *******************************************************************
             * Handle gmmu to sm response
             * *******************************************************************
             */
            for (auto it = warp->gmmu_to_sm_queue.begin(); it != warp->gmmu_to_sm_queue.end();)
            {
                auto access = *it;
                for (auto& thread : warp->mthreads)
                {
                    if (thread.state == Waiting && thread.access == access)
                    {
                        /* Is the last access ? */
                        if (thread.request->writePages.empty())
                        {
                            MemoryAccess* temp = thread.access;
                            thread.access = nullptr;
                            delete temp;
                            thread.state = Idle;

                        } else {

                            thread.state = Busy;
                        }
                        it = warp->gmmu_to_sm_queue.erase(it)++;
                        break;
                    }
                }
                ++it;
            }



            /* *******************************************************************
             * Handle the warp status
             * *******************************************************************
             */
            warp->isBusy = !block->runningKernel->requests.empty();
            for (auto& thread : warp->mthreads)
            {
                warp->isBusy |= !(thread.state == Idle);
            }


            /* *******************************************************************
             * Bind request to idel threads
             * *******************************************************************
             */
            for (auto& thread : warp->mthreads)
            {
                if (thread.state == Idle && !block->runningKernel->requests.empty())
                {
                    thread.request = block->runningKernel->accessRequest();
                    log_V("Executing request", to_string(thread.request->requst_id));
                    thread.readIndex = 0;
                    thread.state = Busy;
                }
            }


            /* *******************************************************************
             * Launch the access
             * *******************************************************************
             */
            for (auto& thread : warp->mthreads)
            {
                if (thread.state != Busy) continue;
                
                /* Handle the read addresses */
                if (thread.readIndex != thread.request->readPages.size()) 
                {
                    thread.access = new MemoryAccess(block->runningKernel->modelID, smID, block->block_id, warp->warpID, thread.request->requst_id, AccessType::Read);
                    
                    for (int i = 0; i < GPU_MAX_ACCESS_NUMBER && thread.readIndex != thread.request->readPages.size(); i++)
                    {
                        auto& page_pair = thread.request->readPages.at(thread.readIndex);
                        thread.access->pageIDs.push_back(page_pair.first);

                        ((--page_pair.second) == 0) && (++thread.readIndex);
                    }
                    ASSERT(thread.access->pageIDs.size() <= GPU_MAX_ACCESS_NUMBER);
                } 

                /* Executing the request */
                else if (thread.request->numOfInstructions-- != 0) {}

                /* Handle the write addresses */ 
                else if (!thread.request->writePages.empty()) 
                {
                    thread.access = new MemoryAccess(block->runningKernel->modelID, smID, block->block_id, warp->warpID, thread.request->requst_id, AccessType::Write);
                    
                    for (int i = 0; i < GPU_MAX_ACCESS_NUMBER && !thread.request->writePages.empty(); i++)
                    {
                        auto& page_pair = thread.request->writePages.front();
                        thread.access->pageIDs.push_back(page_pair.first);
                        
                        if ((--page_pair.second) == 0)
                        {
                            thread.request->writePages.erase(thread.request->writePages.begin());
                        }
                    }
                }

                /* push access to gmmu */
                if (!thread.access->pageIDs.empty())
                {
#if (PRINT_ACCESS_PATTERN)
                    cout << "New access page: ";
                    for (auto page_id : thread.access->pageIDs)
                    {
                        cout << page_id << ", ";
                    }
                    cout << endl;
#endif
                    access_count += thread.access->pageIDs.size();
                    warp->sm_to_gmmu_queue.push_back(thread.access);
                    thread.state = Waiting;
                }
            }

            
        }
        log_V("Total Access Request", to_string(access_count));
    }
}


/** ===============================================================================================
 * \name    bindKernel
 * 
 * \brief   Try to bind kernel into SM according to the resource bound.
 * 
 * \return  return True if bind success
 * 
 * \endcond
 * ================================================================================================
 */
bool
SM::bindKernel(Kernel* kernel)
{
    if (resource.remaining_blocks == 0 || resource.remaining_warps == 0) return false;

    /* Baseline: each kernel get all resource of SM */
    int launch_block_num = ceil(resource.remaining_warps / GPU_MAX_WARP_PER_BLOCK);
    for (int i = 0; i < launch_block_num; i++)
    {
        Block* b = new Block(kernel);

        /* Greedy bind all available warp to current block */
        for (auto& warp : mWarps)
        {
            if(warp.second.isIdle)
            {
                b->warps.push_back(&warp.second);
                warp.second.isIdle = false;
                --resource.remaining_warps;
            }
            if (b->warps.size() == GPU_MAX_WARP_PER_BLOCK) break;
        }
        
#if (LOG_LEVEL >= VERBOSE)
        cout << "Launch kernel:" << kernel->kernelID << " to SM: " << smID << " with warps: " << b->warps.size() << endl;
#endif
        runningBlocks.emplace_back(move(b));
        resource.remaining_blocks--;
    }
    return true;
}


/** ===============================================================================================
 * \name    checkFinish
 * 
 * \brief   Check whether a block is finish
 * 
 * \endcond
 * ================================================================================================
 */
void
SM::checkFinish()
{
    for (auto block = runningBlocks.begin(); block != runningBlocks.end(); ++block)
    {
        (*block)->isFinish = true;
        for (auto& warp : (*block)->warps)
        {
            (*block)->isFinish &= !warp->isBusy;
        }

        if((*block)->isFinish)
        {
            recycleResource(*block);

            delete *block;
            block = runningBlocks.erase(block);
        }
    }
}


/** ===============================================================================================
 * \name    recycleResource
 * 
 * \brief   Re-cycle the used resource on the block.
 * 
 * \param   block   the finished block
 * 
 * \endcond
 * ================================================================================================
 */
void
SM::recycleResource(Block* block)
{
    ASSERT(block->isFinish);

#if (LOG_LEVEL >= VERBOSE)
    cout << "Release block: " << block->block_id << " from SM: " << smID << " with warps: " << block->warps.size() << endl;
#endif

    for (auto& warp : block->warps)
    {
        ++resource.remaining_warps;
        warp->isIdle = true;
    }

    resource.remaining_blocks++;
}


/** ===============================================================================================
 * \name    isComputing
 * 
 * \brief   Check whether the SM is computing or idle in this cycle.
 * 
 * \endcond
 * ================================================================================================
 */
bool
SM::isComputing()
{
    return true;
}


/** ===============================================================================================
 * \name    isRunning
 * 
 * \brief   Check whether there still SM is running.
 * 
 * \endcond
 * ================================================================================================
 */
bool
SM::isRunning()
{
    return !runningBlocks.empty();
}


/** ===============================================================================================
 * \name    checkIsComplete
 * 
 * \brief   Check whether the kernel is complete.
 * 
 * \endcond
 * ================================================================================================
 */
bool
SM::checkIsComplete(Kernel* kernel)
{
    bool complete = true;
    for (auto& block : runningBlocks)
    {
        complete &= !(block->runningKernel == kernel);
    }

    return complete;
}
