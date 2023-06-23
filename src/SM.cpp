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
    for (auto block : runningBlocks)
    {
        log_V("SM", to_string(smID) + " Execute block: " + to_string(block->blockID));

        for (auto warp : block->warps)
        {
            if (!warp->isBusy) continue;
            /* *******************************************************************
             * Handle the warp status
             * *******************************************************************
             */
            bool sync = true;
            for (auto& thread : warp->mthreads) sync &= (thread.state == Idle);
            
            warp->isBusy = !(sync && block->requests.empty());

            /* *******************************************************************
             * Handle gmmu to sm response
             * *******************************************************************
             */
            for (auto access : warp->gmmu_to_warp_queue)
            {
                ASSERT(warp->mthreads.at(access->thread_id).state == Waiting, "Error thread id");
                warp->record.return_access_counter++;
                
                auto& thread = warp->mthreads.at(access->thread_id);

                /* Is the access finish? */
                if (thread.writeIndex != thread.request->writePages.size()) 
                    thread.state = Busy;
                else 
                {
                    thread.state = Idle;
                    delete thread.request;
                }

                delete thread.access;
            }
            warp->gmmu_to_warp_queue.clear();



            /* *******************************************************************
             * Bind request to idel threads by following the SIMT policy
             * *******************************************************************
             */
            if (sync) 
            {
                for (auto& thread : warp->mthreads)
                {
                    if (!block->requests.empty())
                    {
                        thread.request = block->requests.front();
                        block->requests.pop();
                        log_V("Executing request", to_string(thread.request->requst_id));
                        thread.readIndex = 0;
                        thread.state = Busy;
                    } else {
                        break;
                    }
                }
            }
            


            /* *******************************************************************
             * Launch the access
             * *******************************************************************
             */
            for (int i = 0; i < GPU_MAX_THREAD_PER_WARP; i++)
            {
                if (warp->mthreads.at(i).state != Busy) continue;

                auto& thread = warp->mthreads.at(i);
                
                /* ******************************
                 * Handle the read addresses
                 * ******************************
                 */
                if (thread.readIndex != thread.request->readPages.size()) 
                {
                    thread.access = new MemoryAccess(block->runningKernel->appID, block->runningKernel->modelID, smID, block->blockID, warp->warpID, i, thread.request->requst_id, AccessType::Read);
                    
                    for (int i = GPU_MAX_ACCESS_NUMBER; i > 0 && thread.readIndex != thread.request->readPages.size();)
                    {
                        auto& page_pair = thread.request->readPages.at(thread.readIndex);

                        int count = min(page_pair.second, i);
                        thread.access->pageIDs.push_back(page_pair.first);
                        page_pair.second -= count;

                        (page_pair.second == 0) && (++thread.readIndex);
                        i -= count;
                    }
                    ASSERT(thread.access->pageIDs.size() <= GPU_MAX_ACCESS_NUMBER);
                }

                /* ******************************
                 * Executing the request
                 * ******************************
                 */
                else if (thread.request->numOfInstructions-- > 0) {
                    continue;
                }

                /* ******************************
                 * Handle the write addresses
                 * ******************************
                 */
                else if (thread.writeIndex != thread.request->writePages.size()) 
                {
                    thread.access = new MemoryAccess(block->runningKernel->appID, block->runningKernel->modelID, smID, block->blockID, warp->warpID, i, thread.request->requst_id, AccessType::Write);
                    
                    for (int i = GPU_MAX_ACCESS_NUMBER; i > 0 && thread.writeIndex != thread.request->writePages.size();)
                    {
                        auto& page_pair = thread.request->writePages.at(thread.writeIndex);

                        int count = min(page_pair.second, i);
                        thread.access->pageIDs.push_back(page_pair.first);
                        page_pair.second -= count;

                        (page_pair.second == 0) && (++thread.writeIndex);
                        i -= count;
                    }
                    ASSERT(thread.access->pageIDs.size() <= GPU_MAX_ACCESS_NUMBER);
                } 

                /* ******************************
                 * Exception
                 * ******************************
                 */
                else ASSERT(false, "Busy thread should not be idle");

                /* ******************************
                 * push access to gmmu
                 * ******************************
                 */
                warp->record.launch_access_counter++;
                warp->record.access_page_counter += thread.access->pageIDs.size();

                warp->warp_to_gmmu_queue.push_back(thread.access);
                thread.state = Waiting;
                
#if (PRINT_ACCESS_PATTERN)
                std::cout << "New access page: ";
                for (auto page_id : thread.access->pageIDs)
                {
                    std::cout << page_id << ", ";
                }
                std::cout << std::endl;
#endif
            } 
            log_V("Warp ID", to_string(warp->warpID));
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
SM::bindKernel(Kernel* kernel, int num_of_request)
{
    if (resource.remaining_blocks == 0 || resource.remaining_warps == 0) return false;

    /* Baseline: each kernel get all resource of SM */
    int launch_block_num = ceil(resource.remaining_warps / GPU_MAX_WARP_PER_BLOCK);
    for (int i = 0; i < launch_block_num; i++)
    {
        Block* b = new Block(kernel);
        b->record.sm_id = smID;
        b->record.block_id = b->blockID;
        b->record.start_cycle = total_gpu_cycle;

        /* Greedy bind all available warp to current block */
        for (auto& warp : mWarps)
        {
            if(warp.second.isIdle)
            {
                warp.second.isIdle = false;
                warp.second.isBusy = true;
                warp.second.record = {};
                warp.second.record.warp_id = warp.second.warpID;
                warp.second.record.start_cycle = total_gpu_cycle;

                b->warps.push_back(&warp.second);
                b->record.launch_warp_counter++;

                resource.remaining_warps--;
            }
            if (b->warps.size() == GPU_MAX_WARP_PER_BLOCK) break;
        }

        for (int i = 0; i < num_of_request; i++)
        {
            if (!kernel->requests.empty()) b->requests.push(move(kernel->accessRequest()));
        }
        
        
#if (PRINT_SM_ALLCOATION_RESULT)
        std::cout << "Launch kernel:" << kernel->kernelID << " to SM: " << smID << " with warps: " << b->warps.size() << std::endl;
#endif
        runningBlocks.emplace_back(move(b));
        resource.remaining_blocks--;
    }
    return true;
}


/** ===============================================================================================
 * \name    terminateKernel
 * 
 * \brief   Erase the kernel from the SM runningBlocks queue, clear all used resource inlcudes in
 *          Block, Warp, threads
 * 
 * \param   kernel  the kernel pointer going to terminate
 * 
 * \return  true / false of terminate kernel
 * 
 * \endcond
 * ================================================================================================
 */
bool
SM::terminateKernel(Kernel* kernel)
{
    for (auto block : runningBlocks)
    {
        if (block->runningKernel == kernel)
        {
            for (auto warp : block->warps)
            {
                warp->record = {};
                for (auto thread : warp->mthreads)
                {
                    delete thread.request;
                    delete thread.access;
                }
                warp->gmmu_to_warp_queue.clear();
                warp->isBusy = false;
                warp->isIdle = true;

            }
            std::cout << "Release kernel:" << kernel->kernelID << " to SM: " << smID << " with warps: " << block->warps.size() << std::endl;
        }
    }
    runningBlocks.remove_if([kernel](Block* b){return b->runningKernel == kernel;});
    return true;
}


/** ===============================================================================================
 * \name    checkBlockFinish
 * 
 * \brief   Check whether a block is finish
 * 
 * \endcond
 * ================================================================================================
 */
void
SM::checkBlockFinish()
{
    for (auto block = runningBlocks.begin(); block != runningBlocks.end();)
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
            runningBlocks.erase(block++);
        } else {
            block++;
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
#if (PRINT_SM_ALLCOATION_RESULT)
    std::cout << "Release kernel:" << block->runningKernel->kernelID << " from SM: " << smID << " with warps: " << block->warps.size() << std::endl;
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
 * \name    checkKernelComplete
 * 
 * \brief   Check whether the kernel is complete.
 * 
 * \endcond
 * ================================================================================================
 */
bool
SM::checkKernelComplete(Kernel* kernel)
{
    bool complete = true;
    for (auto& block : runningBlocks) complete &= !(block->runningKernel == kernel);

    return complete;
}
