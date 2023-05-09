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
#include "Memory.hpp"
#include "Warp.hpp"

/* ************************************************************************************************
 * Type Define
 * ************************************************************************************************
 */
struct SMInfo {
    unsigned long long exec_cycle = 0;
	unsigned long long computing_cycle = 0;
	unsigned long long wait_cycle = 0;
	unsigned long long idle_cycle = 0;
};

struct ComputingResource {
    int remaining_blocks  = GPU_MAX_BLOCK_PER_SM;
    int remaining_warps   = GPU_MAX_WARP_PER_SM;
    int remaining_threads = GPU_MAX_THREAD_PER_SM;
    int remaining_shmem   = GPU_SHARED_MEMORY_PER_SM;
    int remaining_regs    = GPU_REGISTER_PER_SM;
};


/** ===============================================================================================
 * \name    Block
 * 
 * \brief   The class of ...
 * 
 * \endcond
 * ================================================================================================
 */
class Block
{
/* ************************************************************************************************
 * Class Constructor
 * ************************************************************************************************
 */ 
public:

    Block(Kernel* kernel) : block_id(blockCount++), runningKernel(kernel), isFinish(false) {};

/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
public:
    int block_id;

    bool isFinish;

	unsigned launch_warp_counter = 0;
	unsigned long long launch_access_counter = 0; 
	unsigned long long return_access_counter = 0;

	Kernel* runningKernel = nullptr;

    list<Warp*> warps;

private:
    /* Number of block be created */
    static int blockCount;
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
 * Functions
 * ************************************************************************************************
 */
public:
    void cycle ();

    bool bindKernel(Kernel* kernel);
    void recycleResource(Block* block);
    void checkFinish();

    bool isComputing();
    bool isRunning();
    bool checkIsComplete(Kernel* kernel);

    void setGMMU (GMMU* gmmu) {mGMMU = gmmu;}
    
    ComputingResource getResourceInfo() const {return resource;}
/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
public:
    const int smID;

private:
    static int SMCount;

    GMMU* mGMMU;

    SMInfo info;

    map<int, Warp> mWarps;

    ComputingResource resource;

    list<Block*> runningBlocks;

friend GMMU;
};

#endif