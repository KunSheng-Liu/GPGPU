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

    GMMU(GPU* gpu);

   ~GMMU();

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    void cycle ();

    void Access_Processing ();
    void Page_Fault_Handler ();
    
/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
	queue<MemoryAccess*> sm_to_gmmu_queue;
	queue<MemoryAccess*> gmmu_to_sm_queue;
private:

    GPU* mGPU;

    /* *******************************************************************
     * \param model_id      the cgroup is isolated in each model
     * \param cgroup        the cgroup, use LRU
     * *******************************************************************
    */
	map<int, pair<int, LRU_TLB<int, Page*>>> mCGroups;

};

#endif