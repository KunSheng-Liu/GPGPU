/** 
 * \name    Scheduler.hpp
 * 
 * \brief   Declare the approach function pointer
 * 
 * \date    May 25, 2023
 */
#ifndef _SCHEDULER_HPP_
#define _SCHEDULER_HPP_

#include "App_config.h"

#include "Application.hpp"
#include "CPU.hpp"
#include "Kernel.hpp"

class Inference_Admission_API;
class Kernel_Scheduler_API;
class Memory_Allocator_API;

/** ===============================================================================================
 * \name    Scheduler
 * 
 * \brief   The base class of NN scheduler. You can add new schedule policy by adding them into 
 *          Inference_Admission_API | Kernel_Scheduler_API | Memory_Allocator_API
 * 
 * \note    * This defualt scheduler didn't block SM to any application, just build tasks into model
 *          and launch all ready kernels to GPU's commandQueue. Which leads the kernel contension
 *          the SM resource in GPU core.
 * 
 * \note    * All application can run one model once a time
 * 
 * \endcond
 * ================================================================================================
 */
class Scheduler
{
/* ************************************************************************************************
 * Class Constructor
 * ************************************************************************************************
 */ 
public:
    Scheduler (CPU* cpu);

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    void Sched ();

private:
    void missDeadlineHandler ();

/* ************************************************************************************************
 * CallBack Functions
 * ************************************************************************************************
 */
private:
    bool (*Inference_Admission) (CPU* mCPU);
    bool (*Kernel_Scheduler)    (CPU* mCPU);
    bool (*Memory_Allocator)    (CPU* mCPU);

/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
private:
    CPU* mCPU;
};

class Inference_Admission_API
{
public:
    static bool Baseline (CPU* mCPU);
    static bool Greedy   (CPU* mCPU);
    static bool BARM     (CPU* mCPU);
    static bool LazyB    (CPU* mCPU);
    static bool My       (CPU* mCPU);
};

class Kernel_Scheduler_API
{
public:
    static bool Baseline (CPU* mCPU);
    static bool LazyB    (CPU* mCPU);
};

class Memory_Allocator_API
{
public:
    static bool None    (CPU* mCPU);
    static bool Average (CPU* mCPU);
    static bool MEMA    (CPU* mCPU);
    static bool R_MEMA  (CPU* mCPU);
};

#endif