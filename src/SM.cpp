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

        int access_count = 0;
        for (auto& warp : block->warps)
        {
            /* *******************************************************************
             * Handle gmmu to sm response
             * *******************************************************************
             */
            for (auto access = warp->gmmu_to_sm_queue.begin(); access != warp->gmmu_to_sm_queue.end();)
            {
                for (auto thread = warp->waitingThread.begin(); thread != warp->waitingThread.end(); thread++)
                {
                    if (thread->access == *access) 
                    {
                        MemoryAccess* temp = thread->access;
                        thread->access = nullptr;
                        delete temp;

                        /* Is the request finish ? */
                        if (thread->request->writePages.empty())
                        {
                            Request* temp = thread->request;
                            thread->request = nullptr;
                            delete temp;
                            warp->idleThread.splice(warp->idleThread.end(), warp->waitingThread, thread);
                        } else {
                            warp->busyThread.splice(warp->busyThread.end(), warp->waitingThread, thread);
                        }
                        access = warp->gmmu_to_sm_queue.erase(access)++;
                        break;
                    }
                }
                ASSERT("Handle gmmu to sm", "Missing Access in waiting thread");
            }


            /* *******************************************************************
             * Handle the warp status
             * *******************************************************************
             */
            if (warp->idleThread.size() == GPU_MAX_THREAD_PER_WARP)
            {
                warp->isBusy = !block->runningKernel->requests.empty();
            }
            
            /* *******************************************************************
             * Bind request to idel threads
             * *******************************************************************
             */
            for (auto thread = warp->idleThread.begin(); thread != warp->idleThread.end() && !block->runningKernel->requests.empty();)
            {
                thread->request = block->runningKernel->accessRequest();
                thread->readIndex = 0;
                log_D("Executing request", to_string(thread->request->requst_id));
                warp->busyThread.splice(warp->busyThread.end(), warp->idleThread, thread++);
            }


            /* *******************************************************************
             * Launch the access
             * *******************************************************************
             */
            for (auto thread = warp->busyThread.begin(); thread != warp->busyThread.end();)
            {
                /* Handle the read addresses */
                if (thread->readIndex != thread->request->readPages.size()) 
                {
                    thread->access = new MemoryAccess(block->runningKernel->modelID, smID, block->block_id, warp->warpID, thread->request->requst_id, AccessType::Write);
                    
                    auto& page_pair = thread->request->readPages.at(thread->readIndex);
                    thread->access->page_id = page_pair.first;

                    ((--page_pair.second) == 0) && (++thread->readIndex);
                } 

                /* Executing the request */
                else if (thread->request->numOfInstructions-- != 0) {}

                /* Handle the write addresses */ 
                else if (!thread->request->writePages.empty()) 
                {
                    thread->access = new MemoryAccess(block->runningKernel->modelID, smID, block->block_id, warp->warpID, thread->request->requst_id, AccessType::Write);
                    
                    auto& page_pair = thread->request->writePages.front();
                    thread->access->page_id = page_pair.first;
                    if ((--page_pair.second) == 0)
                    {
                        thread->request->writePages.erase(thread->request->writePages.begin());
                    }
                }

                /* push access to gmmu */
                if (!(thread->access == nullptr))
                {
                    ++access_count;
                    warp->sm_to_gmmu_queue.push_back(thread->access);
                    warp->waitingThread.splice(warp->waitingThread.end(), warp->busyThread, thread++);

                } else {
                    ++thread;
                }
                
            }
            
        }
        cout << "Total Access Request: " << access_count << endl;
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
        
        cout << "Launch kernel:" << kernel->kernelID << " to SM: " << smID << " with warps: " << b->warps.size() << endl;

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

    cout << "Release block: " << block->block_id << " from SM: " << smID << " with warps: " << block->warps.size() << endl;

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
