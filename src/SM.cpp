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
    for (auto block : runningBlocks)
    {
        cout << "Execute block: " << block->block_id << endl;

        for (int i = 0; i < block->bind_thread_number; i++)
        {
            if(block->running_kernel->requests.empty())
            {
                block->finish = true;
                block->end_cycle = total_gpu_cycle;
                break;
            }
            
            Request* request = block->running_kernel->accessRequest();
        }
    }

    runningBlocks.remove_if([](Block* b){return b->finish;});

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
    if (reousrceInfo.remaining_threads == 0) return false;

    kernel->record.block_info = new Block(kernel, total_gpu_cycle);
    
    Block* block = kernel->record.block_info;

    block->block_id = kernel->kernelID;

    block->bind_thread_number = min(GPU_MAX_THREAD_PER_BLOCK, reousrceInfo.remaining_threads);

    reousrceInfo.remaining_threads -= min(GPU_MAX_THREAD_PER_BLOCK, reousrceInfo.remaining_threads);

    runningBlocks.push_back(block);

    kernel->record.running_sm = this;

    log_D("bindKernel", "kernel: " + to_string(kernel->kernelID) + " bind to SM: " + to_string(smID));

    return true;
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

    reousrceInfo.remaining_threads += block->bind_thread_number;
    runningBlocks.remove(block);
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
    return false;
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
SM::checkIsComplete()
{
    return false;
}
