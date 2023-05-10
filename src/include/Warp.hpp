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
struct AccessThread {
    /* Read index for avoiding the erase overhead */
    int readIndex = 0;

    Request* request;
    MemoryAccess* access;
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

    Warp(int id = -1) : warpID(id), idleThread(list<AccessThread>(GPU_MAX_THREAD_PER_WARP)), busyThread({})
           , isIdle(true), isBusy(false) {} 

   ~Warp() {}

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

    /* The thread queues that handle the access state machine */
    list<AccessThread> idleThread;      // idel
    list<AccessThread> busyThread;      // is executing
    list<AccessThread> waitingThread;   // is waiting gmmu handle
    
	list<MemoryAccess*> sm_to_gmmu_queue;
	list<MemoryAccess*> gmmu_to_sm_queue;

friend GMMU;
};


#endif