/**
 * \name    Memory.hpp
 * 
 * \brief   Declare the memory API 
 * 
 * \date    APR 4, 2023
 */

#ifndef _MEMORY_HPP_
#define _MEMORY_HPP_

/* ************************************************************************************************
 * Include Library
 * ************************************************************************************************
 */
#include "App_config.h"
#include "Log.h"


/* ************************************************************************************************
 * Type Define
 * ************************************************************************************************
 */ 
typedef enum {
	Write = 0,
	Read  = 1,
}AccessType;

struct MemoryAccess {
    int model_id;
	int sm_id;
	int block_id;
	int warp_id;
    int request_id;
    AccessType type;

    vector<unsigned long long> pageIDs = {};

    MemoryAccess(int model_id, int sm_id, int block_id, int warp_id, int request_id, AccessType type) 
            : model_id(model_id), sm_id(sm_id), block_id(block_id), warp_id(warp_id), request_id(request_id), type(type) {}
};

/** ===============================================================================================
 * \name    Memory
 * 
 * \brief   The base class of the memory hierarchy. You can implement the cache, RAM by inheritance 
 *          this class.
 * \endcond
 * ================================================================================================
 */
class Memory
{
/* ************************************************************************************************
 * Class Constructor
 * ************************************************************************************************
 */ 
public:

    Memory(int size); // size in Byte

   ~Memory();

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    int dataWrite (int PA, int8_t data);
    int8_t dataRead (int PA);



/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
public:

    /* The storage size of memory */
    const int storageSize; 	                // unit (KB)

    /* The throughput size  */
    int dataWidth_I;
    int dataWidth_O;

    /* The clock speed */
    int clockSpeed;

    /* The actually storaged data */
    unordered_map<int, unsigned char> data;    // first is PA, second is data

    /* Recorder */
    int numOfRead;
    int numOfWrite;

};

#endif