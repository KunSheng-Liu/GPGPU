/**
 * \name    SM.hpp
 * 
 * \brief   Declare the structure of SM
 * 
 * \date    May 7, 2023
 */

#ifndef _SM_HPP_
#define _SM_HPP_

/* ************************************************************************************************
 * Include Library
 * ************************************************************************************************
 */
#include "App_config.h"
#include "Log.h"

#include "Kernel.hpp"
#include "Block.hpp"
#include "Memory.hpp"
#include "Warp.hpp"

/* ************************************************************************************************
 * Type Define
 * ************************************************************************************************
 */
struct SM_Resource {
    int remaining_blocks  = GPU_MAX_BLOCK_PER_SM;
    int remaining_warps   = GPU_MAX_WARP_PER_SM;
    int remaining_threads = GPU_MAX_THREAD_PER_SM;
    int remaining_shmem   = GPU_SHARED_MEMORY_PER_SM;
    int remaining_regs    = GPU_REGISTER_PER_SM;
};


/** ===============================================================================================
 * \name    SM
 * 
 * \brief   Contains the model and it's data
 * 
 * \endcond
 * ================================================================================================
 */
class SM
{
/* ************************************************************************************************
 * Class Constructor
 * ************************************************************************************************
 */ 
public:

    SM();

   ~SM();

/* ************************************************************************************************
 * Type Define
 * ************************************************************************************************
 */
struct SMRecord {
    unsigned long long start_cycle = 0, end_cycle = 0;
    unsigned long long exec_cycle = 0, idle_cycle = 0;
};

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    void cycle ();

    bool bindKernel(Kernel* kernel, int num_of_request);
    bool terminateKernel (Kernel* kernel);
    void checkBlockFinish();

    bool isComputing();
    bool isIdel();
    bool checkKernelComplete(Kernel* kernel);

    void setGMMU (GMMU* gmmu) {mGMMU = gmmu;}
    
    SM_Resource getResourceInfo() const {return resource;}

private:
    void recycleResource(Block* block);
    void statistic();

/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
public:
    const int smID;

private:
    static int SMCount;

    GMMU* mGMMU;

    SMRecord record;

    map<int, Warp> mWarps;

    SM_Resource resource;

    list<Block*> runningBlocks;

friend GMMU;
};

#endif