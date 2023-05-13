/**
 * \name    Block.hpp
 * 
 * \brief   Declare the structure of Block
 * 
 * \date    May 14, 2023
 */

#ifndef _BLOCK_HPP_
#define _BLOCK_HPP_

/* ************************************************************************************************
 * Include Library
 * ************************************************************************************************
 */
#include "App_config.h"
#include "Log.h"

#include "Kernel.hpp"
#include "Warp.hpp"

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

    Block(Kernel* kernel) : block_id(blockCount++), runningKernel(kernel) {}

   ~Block() {}

/* ************************************************************************************************
 * Type Define
 * ************************************************************************************************
 */
struct BlockRecord {
    int block_id, sm_id;
    
	unsigned launch_warp_counter = 0;
    unsigned long long start_cycle = 0, end_cycle = 0;
    unsigned long long access_page_counter = 0;
	unsigned long long launch_access_counter = 0; 
	unsigned long long return_access_counter = 0;
    
    list<Warp::WarpRecord> warp_record;

    BlockRecord(int block_id = -1) : block_id(block_id) {};
};

/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
public:
    int block_id;

    BlockRecord record;

    list<Warp*> warps;

	Kernel* runningKernel = nullptr;

private:
    /* Number of block be created */
    static int blockCount;
};


#endif