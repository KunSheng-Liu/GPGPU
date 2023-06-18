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
	Read,
	Write,
} AccessType;

struct MemoryAccess {
    int app_id;
    int model_id;
	int sm_id;
	int block_id;
	int warp_id;
    int thread_id;
    int request_id;
    AccessType type;

    vector<unsigned long long> pageIDs = {};

    MemoryAccess(int app_id, int model_id, int sm_id, int block_id, int warp_id, int thread_id, int request_id, AccessType type) 
            : app_id(app_id), model_id(model_id), sm_id(sm_id), block_id(block_id), warp_id(warp_id), thread_id(thread_id), request_id(request_id), type(type) {}
};

/* All avaliable memory type */
typedef enum {
    SPACE_NONE,
    SPACE_VRAM,
	SPACE_DRAM,
    // ReRAM,
    // NVME,
    // HDD,
    // ...
} Memory_t;

struct IO_Channel {
    int waiting_cycle;
    MemoryAccess* access;

    IO_Channel() : waiting_cycle(0), access(nullptr) {}
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

    Memory(Memory_t memory_type = SPACE_NONE, unsigned long long storage_size = 0, int total_bandwidth = 128, int channel_bandwidth = 32);

   ~Memory();

/* ************************************************************************************************
 * Type Define
 * ************************************************************************************************
 */
struct MemoryRecord {
    unsigned long long idle_cycle = 0;
    unsigned long long exec_cycle = 0;

    unsigned long long numOfRead  = 0;
    unsigned long long numOfWrite = 0;
};

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    virtual void cycle ();

    virtual bool Read  (int num_of_bytes, MemoryAccess* access);
    virtual bool Write (int num_of_bytes, MemoryAccess* access);

/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
public:
    /* The index of memory. Each memory have a unique index */
    const int memoryIndex;

    /* The type of memory */
    Memory_t memoryType;

    /* The storage size of memory */
    const unsigned long long storageSize;          // unit (Byte)
    const unsigned long long startPhysicalAddress; // unit (Byte)
    const unsigned long long endPhysicalAddress;   // unit (Byte)

    const unsigned totalBandwidth;
    const unsigned channelBandwidth;

private:
    /* Number of memory be created */
    static int memoryCount;
    static unsigned long long storageCount;

protected:
    /* Recorder */
    MemoryRecord recorder;

    map<int, IO_Channel> IO_Channels;
    list<int> idleChannelList;

    /* The actually storaged data */
    unordered_map<unsigned long long, unsigned int> storage;     // first is PA, second is data
    
    unordered_set<MemoryAccess*>  access_finish_queue;

friend MemoryController;
};


class DRAM : public Memory
{
public:
/* ************************************************************************************************
 * Class Constructor
 * ************************************************************************************************
 */ 
public:
    DRAM(unsigned long long storage_size);

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:

};

#endif