/**
 * \name    SM.cpp
 * 
 * \brief   Implement the SM and it's components.
 * 
 * \date    APR 30, 2023
 */

#include "include/SM.hpp"

/* ************************************************************************************************
 * Global Variable
 * ************************************************************************************************
 */
int SM::SMCount = 0;

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

        for (int i = 0; i < block->bind_warp_number && !block->runningKernel->requests.empty(); i++)
        {
            for (int j = 0; j < GPU_THREAD_PER_WARP && !block->runningKernel->requests.empty(); j++)
            {
                Request* request = block->runningKernel->accessRequest();

                // for (auto read : request->readPages)
                // {
                //     if (read.second > 1)
                //         cout << "Read: " << read.first << " , " << read.second << " times" << endl;
                // }

                delete request;
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

    b->block_id = kernel->kernelID;
    b->bind_warp_number = resource.remaining_warps;

    cout << "Launch kernel:" << kernel->kernelID << " to SM: " << smID << " with warps: " << b->bind_warp_number << endl;

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

    cout << "Release block: " << block->block_id << " from SM: " << smID << " with warps: " << block->bind_warp_number << endl;
    resource.remaining_warps += block->bind_warp_number;
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
