/**
 * \name    CPU.hpp
 * 
 * \brief   Declare the structure of CPU
 * 
 * \note    In this simulator, the VA is the pointer of the data.
 * 
 * \date    APR 6, 2023
 */

#ifndef _CPU_HPP_
#define _CPU_HPP_

/* ************************************************************************************************
 * Include Library
 * ************************************************************************************************
 */
#include "App_config.h"
#include "Log.h"

#include "Application.hpp"
#include "MemoryController.hpp"
#include "MMU.hpp"
#include "Models.hpp"
#include "Scheduler.hpp"

/** ===============================================================================================
 * \name    CPU
 * 
 * \brief   The class of the CPU, contains MMU, TLB.
 * 
 * \endcond
 * ================================================================================================
 */
class CPU
{
/* ************************************************************************************************
 * Class Constructor
 * ************************************************************************************************
 */ 
public:

    CPU(MemoryController* mc, GPU* gpu);

   ~CPU();

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    void cycle ();
    bool Check_All_Applications_Finish();

private:
    bool Check_Finish_Kernel ();

/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
public:
    GPU* mGPU;
    MemoryController* mMC;

    MMU mMMU;
    Scheduler* mScheduler;

    vector<Application*> mAPPs;
};

#endif