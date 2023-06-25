/**
 * \name    GMMU.hpp
 * 
 * \brief   Declare the structure of GMMU
 * 
 * \date    May 9, 2023
 */

#ifndef _GMMU_HPP_
#define _GMMU_HPP_

/* ************************************************************************************************
 * Include Library
 * ************************************************************************************************
 */
#include "App_config.h"
#include "Log.h"

#include "GPU.hpp"
#include "Memory.hpp"
#include "MemoryController.hpp"
#include "TLB.hpp"

/** ===============================================================================================
 * \name    GMMU
 * 
 * \brief   Contains the model and it's data
 * 
 * \endcond
 * ================================================================================================
 */
class GMMU
{
/* ************************************************************************************************
 * Class Constructor
 * ************************************************************************************************
 */ 
public:

    GMMU(GPU* gpu, MemoryController* mc);

   ~GMMU();

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    void cycle ();

    bool terminateModel (int app_id, int model_id);

    void setCGroupSize (int app_id, unsigned long long capacity);
    void freeCGroup (int app_id);
    LRU_TLB<unsigned long long, Page*>* getCGroup (int model_id);

private:
    void Access_Processing ();
    void Page_Fault_Handler ();
    
/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
private:

    GPU* mGPU;
    MemoryController* mMC;

    list<MemoryAccess*> MSHRs;

	list<MemoryAccess*> warps_to_gmmu_queue;
	list<MemoryAccess*> gmmu_to_warps_queue;
    
    /* Page Fault handler */
	long long wait_cycle = 0;
    map<MemoryAccess*, int> access_count;
    list<pair<unsigned long long, list<MemoryAccess*>>> page_fault_process_queue;

    /* *******************************************************************
     * \param model_id      the cgroup is isolated in each model
     * \param cgroup        the cgroup, use LRU
     * *******************************************************************
     */
	map<int, LRU_TLB<unsigned long long, Page*>> mCGroups;

friend SM;
};

#endif