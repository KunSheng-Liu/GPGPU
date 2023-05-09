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
        cout << "SM: " << smID << " Execute block: " << block->block_id << endl;

        for (auto& warp : block->mWarps)
        {
            /* Handle the warp status */
            if (warp.idleThread.size() == GPU_MAX_THREAD_PER_WARP)
            {
                warp.isBusy = false;
                warp.isIdle = true;
                Request* temp = warp.request;
                warp.request  = nullptr;
                delete temp;

                if (!block->runningKernel->requests.empty())
                {
                    /* Start a new request */
                    warp.request = block->runningKernel->accessRequest();
                    log_V("Executing request", to_string(warp.request->requst_id));
                    warp.isIdle = false;
                    warp.isBusy = true;
                } else {
                    /* Finish */
                    block->finish = true;
                }
            }
            
            /* Handling the executing threads */
            for (auto thread = warp.busyThread.begin(); thread != warp.busyThread.end();)
            {
                if (thread->waiting_time == 0) {
                    MemoryAccess* temp = thread->access;
                    thread->access = nullptr;
                    delete temp;
                    warp.idleThread.splice(warp.idleThread.end(), warp.busyThread, thread++);
                } else {
                    thread->waiting_time--;
                    ++thread;
                }
            }

            /* Handle gmmu to sm response */
            for (auto access = gmmu_to_sm_access.begin(); access != gmmu_to_sm_access.end() && !warp.waitingThread.empty();)
            {
                if ((*access)->block_id == block->block_id && (*access)->request_id == warp.request->requst_id)
                {
                    for (auto thread = warp.waitingThread.begin(); thread != warp.waitingThread.end(); thread++)
                    {
                        if (thread->access == *access) {
                            if (warp.request->readPages.empty() && warp.request->writePages.empty())
                            {
                                thread->waiting_time = warp.request->numOfInstructions;
                                warp.busyThread.splice(warp.busyThread.end(), warp.waitingThread, thread);
                            } else {
                                MemoryAccess* temp = thread->access;
                                thread->access = nullptr;
                                delete temp;
                                warp.idleThread.splice(warp.idleThread.end(), warp.waitingThread, thread);
                            }
                            access = gmmu_to_sm_access.erase(access)++;
                            break;
                        }
                    }
                    ASSERT("Handle gmmu to sm", "Missing Access in waiting thread");
                } else {
                    ++access;
                }
            }
            

            /* Launch new access */
            for (auto thread = warp.idleThread.begin(); thread != warp.idleThread.end();)
            {
                /* Handle the read addresses */
                if (!warp.request->readPages.empty()) 
                {
                    thread->access = new MemoryAccess(smID, block->block_id, warp.request->requst_id, AccessType::Read);
                    
                    while (thread->access->page_id.size() < GPU_MAX_ACCESS_NUMBER && !warp.request->readPages.empty())
                    {
                        thread->access->page_id.emplace_back(warp.request->readPages.front().first);
                        if ((--warp.request->readPages.front().second) == 0) 
                        {
                            warp.request->readPages.erase(warp.request->readPages.begin());
                        }
                    }
                } 

                /* Handle the write addresses */ 
                else if (!warp.request->writePages.empty()) 
                {
                    thread->access = new MemoryAccess(smID, block->block_id, warp.request->requst_id, AccessType::Write);

                    while (thread->access->page_id.size() < GPU_MAX_ACCESS_NUMBER && !warp.request->writePages.empty())
                    {
                        thread->access->page_id.emplace_back(warp.request->writePages.front().first);
                        if ((--warp.request->writePages.front().second) == 0) 
                        {
                            warp.request->writePages.erase(warp.request->writePages.begin());
                        }
                    }
                } else {
                    break;
                }

                /* push access to gmmu */
                if (!thread->access->page_id.empty())
                {
                    // cout << "New access page: ";
                    // for (auto page : thread->access->page_id)
                    // {
                    //     cout << page << ", ";
                    // }
                    // cout << endl;
                    sm_to_gmmu_access.push_back(thread->access);
                    warp.waitingThread.splice(warp.waitingThread.end(), warp.idleThread, thread++);

                } else {
                    ++thread;
                }
            }
        }
       
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
    Block* b = new Block(kernel);

    for (int i = 0; i < resource.remaining_warps; i++)
    {
        b->mWarps.emplace_back();
    }
    
    cout << "Launch kernel:" << kernel->kernelID << " to SM: " << smID << " with warps: " << b->mWarps.size() << endl;

    runningBlocks.emplace_back(move(b));
    resource.remaining_warps = 0;
    resource.remaining_blocks--;


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
        if((*block)->runningKernel->requests.empty())
        {
            (*block)->finish = true;
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
    ASSERT(block->finish);

    cout << "Release block: " << block->block_id << " from SM: " << smID << " with warps: " << block->mWarps.size() << endl;

    resource.remaining_warps += block->mWarps.size();
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
