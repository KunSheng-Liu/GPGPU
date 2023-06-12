/**
 * \name    Warp.hpp
 * 
 * \brief   Declare the structure of Warp
 * 
 * \date    May 9, 2023
 */

#ifndef _WARP_HPP_
#define _WARP_HPP_

/* ************************************************************************************************
 * Include Library
 * ************************************************************************************************
 */
#include "App_config.h"
#include "Log.h"

#include "Kernel.hpp"
#include "Memory.hpp"

/* ************************************************************************************************
 * Type Define
 * ************************************************************************************************
 */
typedef enum {
    Idle, 
	Busy, 
    Waiting
}Thread_State;

struct AccessThread {

    int readIndex = 0;   // Read index for avoiding the erase overhead
    int writeIndex = 0;  // Write index for avoiding the erase overhead

    Request* request;
    MemoryAccess* access;

    Thread_State state = Idle;
};


/** ===============================================================================================
 * \name    Warp
 * 
 * \brief   The class of ...
 * 
 * \endcond
 * ================================================================================================
 */
class Warp
{

/* ************************************************************************************************
 * Class Constructor
 * ************************************************************************************************
 */ 
public:

    Warp(int id = -1) : warpID(id), mthreads(vector<AccessThread>(GPU_MAX_THREAD_PER_WARP))
           , isIdle(true), isBusy(false) {} 

   ~Warp() {}

/* ************************************************************************************************
 * Type Define
 * ************************************************************************************************
 */
struct WarpRecord {
    int warp_id;

    unsigned long long start_cycle = 0, end_cycle = 0;
	unsigned long long computing_cycle = 0, wait_cycle = 0;

    unsigned long long access_page_counter = 0;
	unsigned long long launch_access_counter = 0; 
	unsigned long long return_access_counter = 0;
};

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */

/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
public:
    const int warpID;

    bool isIdle;
    bool isBusy;

    WarpRecord record;

    /* The thread queues that handle the access state machine */
    vector<AccessThread> mthreads;
    
	list<MemoryAccess*> gmmu_to_sm_queue;

friend GMMU;
};


#endif