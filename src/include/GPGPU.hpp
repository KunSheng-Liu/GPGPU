/**
 * \name    GPGPU.hpp
 * 
 * \brief   Declare the structure of GPGPU
 * 
 * \note    It's a simulator for simulating the behavior of a GPGPU (General Porpose GPU).
 * 
 * \date    APR 18, 2023
 */

#ifndef _GPGPU_HPP_
#define _GPGPU_HPP_

/* ************************************************************************************************
 * Include Library
 * ************************************************************************************************
 */
#include "App_config.h"
#include "Log.h"
#include "Macro.h"

#include "CPU.hpp"
#include "MemoryController.hpp"

/* ************************************************************************************************
 * Declaration
 * ************************************************************************************************
 */
#define  CPU_MASK       0x01
#define  MC_MASK        0x02 
#define  GPU_MASK       0x04
#define  GMMU_MASK      0x08

#define  cpu_period     double(1.0 / CPU_F);
#define  mc_period      double(1.0 / MC_F);
#define  gpu_period     double(1.0 / GPU_F);
#define  gmmu_period    double(1.0 / GMMU_F);

/** ===============================================================================================
 * \name    GPGPU
 * 
 * \brief   The class of the CPU, contains MMU, TLB.
 * 
 * \endcond
 * ================================================================================================
 */
class GPGPU {
/* ************************************************************************************************
 * Class Constructor
 * ************************************************************************************************
 */ 
public:

    GPGPU();

   ~GPGPU();

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    void run ();

    int next_clock_domain();

/* ************************************************************************************************
 * Module
 * ************************************************************************************************
 */
private:
    MemoryController mMC;
    CPU mCPU;
    GPU mGPU;
    GMMU* mGMMU;

/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
private:
    long double cpu_time  = double(1.0 / CPU_F);
    long double mc_time   = double(1.0 / MC_F);
    long double gpu_time  = double(1.0 / GPU_F);
    long double gmmu_time = double(1.0 / GMMU_F);
};

#endif
