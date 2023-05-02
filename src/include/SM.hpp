/**
 * \name    SM.hpp
 * 
 * \brief   Declare the structure of SM
 * 
 * \date    APR 30, 2023
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
    int remaining_threads = GPU_MAX_THREAD_PER_SM;
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

    Block(Kernel* kernel,  unsigned long long start_cycle) : running_kernel(kernel), start_cycle(start_cycle), finish(false) {};

/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
public:
    int block_id;

    bool finish;

    unsigned long long start_cycle = 0;
    unsigned long long end_cycle = 0;

    unsigned bind_thread_number = 0;

	unsigned long long launch_access_counter = 0;
	unsigned long long return_access_counter = 0;
	unsigned launch_warp_counter = 0;

	Kernel* running_kernel;

	list<int> wait_computing_time;
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

    bool isComputing();
    bool isRunning();
    bool checkIsComplete();

    void setGMMU (GMMU* gmmu) {mGMMU = gmmu;}
    
    ComputingResource getResourceInfo() const {return reousrceInfo;}
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

    ComputingResource reousrceInfo;

    list<Block*> runningBlocks;

};

#endif