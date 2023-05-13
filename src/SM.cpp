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
    log_T("SM " + to_string(smID) + " Cycle", to_string(total_gpu_cycle));

    /* Computing */
    for (auto& block : runningBlocks)
    {
        log_V("SM", to_string(smID) + " Execute block: " + to_string(block->block_id));

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
                        warp->record.return_access_counter++;
                        delete thread.access;

                        /* Is the last access ? */
                        if (!thread.request->writePages.empty()) 
                            thread.state = Busy;
                        else 
                        {
                            delete thread.request;
                            thread.state = Idle;
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
            if (!warp->isBusy) continue;


            /* *******************************************************************
             * Bind request to idel threads by following the SIMT policy
             * *******************************************************************
             */
            bool sync = true;
            for (auto& thread : warp->mthreads) sync &= (thread.state == Idle);

            if (sync) {
                for (auto& thread : warp->mthreads)
                {
                    if (!block->runningKernel->requests.empty())
                    {
                        thread.request = block->runningKernel->accessRequest();
                        log_V("Executing request", to_string(thread.request->requst_id));
                        thread.readIndex = 0;
                        thread.state = Busy;
                    }
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
                else if (thread.request->numOfInstructions-- != 0) {
                    continue;
                }
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
                } else {
                    continue;
                }

                /* push access to gmmu */
                warp->record.launch_access_counter++;
                warp->record.access_page_counter += thread.access->pageIDs.size();

                warp->sm_to_gmmu_queue.push_back(thread.access);
                thread.state = Waiting;
                
#if (PRINT_ACCESS_PATTERN)
                std::cout << "New access page: ";
                for (auto page_id : thread.access->pageIDs)
                {
                    std::cout << page_id << ", ";
                }
                std::cout << endl;
#endif
            } 
            log_V("Total Access", to_string(warp->record.launch_access_counter));
            log_V("Total Access Pages", to_string(warp->record.access_page_counter));
        }
    }

    /* SM statistic */
    statistic();
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
        b->record.sm_id = smID;
        b->record.block_id = b->block_id;
        b->record.start_cycle = total_gpu_cycle;

        /* Greedy bind all available warp to current block */
        for (auto& warp : mWarps)
        {
            if(warp.second.isIdle)
            {
                warp.second.isIdle = false;
                warp.second.record = Warp::WarpRecord();
                warp.second.record.warp_id = warp.second.warpID;
                warp.second.record.start_cycle = total_gpu_cycle;

                b->warps.push_back(&warp.second);
                b->record.launch_warp_counter++;

                resource.remaining_warps--;
            }
            if (b->warps.size() == GPU_MAX_WARP_PER_BLOCK) break;
        }
        
#if (LOG_LEVEL >= VERBOSE)
        std::cout << "Launch kernel:" << kernel->kernelID << " to SM: " << smID << " with warps: " << b->warps.size() << endl;
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
    for (auto block = runningBlocks.begin(); block != runningBlocks.end(); block++)
    {
        bool finish = true;
        for (auto& warp : (*block)->warps) finish &= !warp->isBusy;

        if(finish)
        {
            /* Passing the runtime record */
            for (auto& warp : (*block)->warps) 
            {
                warp->record.end_cycle = total_gpu_cycle;
                (*block)->record.access_page_counter += warp->record.access_page_counter;
                (*block)->record.launch_access_counter += warp->record.launch_access_counter;
                (*block)->record.return_access_counter += warp->record.return_access_counter; 
                (*block)->record.warp_record.push_back(move(warp->record));
            }
            ASSERT((*block)->record.launch_access_counter == (*block)->record.return_access_counter, "Block finish error");
            
            (*block)->record.end_cycle = total_gpu_cycle;
            (*block)->runningKernel->block_record.push_back(move((*block)->record));
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
#if (LOG_LEVEL >= VERBOSE)
    std::cout << "Release block: " << block->block_id << " from SM: " << smID << " with warps: " << block->warps.size() << endl;
#endif

    for (auto& warp : block->warps)
    {
        resource.remaining_warps++;
        warp->isIdle = true;
    }

    resource.remaining_blocks++;
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
SM::statistic()
{
    isIdel() ? record.exec_cycle++ : record.idle_cycle++;

    for (auto& warp : mWarps)
    {
        warp.second.isBusy ? warp.second.record.computing_cycle++ : warp.second.record.wait_cycle++;
    }
}


/** ===============================================================================================
 * \name    isComputing
 * 
 * \brief   Check whether the SM is computing or waiting in this cycle.
 * 
 * \endcond
 * ================================================================================================
 */
bool
SM::isComputing()
{
    bool Computing = true;
    for (auto& warp : mWarps) Computing &= warp.second.isBusy;

    return Computing;
}


/** ===============================================================================================
 * \name    isIdel
 * 
 * \brief   Check whether there still SM is idle.
 * 
 * \endcond
 * ================================================================================================
 */
bool
SM::isIdel()
{
    bool idle = true;
    for (auto& warp : mWarps) idle &= warp.second.isIdle;

    return idle;
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
    for (auto& block : runningBlocks) complete &= !(block->runningKernel == kernel);

    return complete;
}
