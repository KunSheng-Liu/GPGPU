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
    virtual void Sched () = 0;

protected:
    void kernelLauncher (Kernel* kernel);
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
protected:
    CPU* mCPU;
};

/** ===============================================================================================
 * \name    Scheduler_Baseline
 * 
 * \brief   ...
 * 
 * \endcond
 * ================================================================================================
 */
class Scheduler_Baseline : public Scheduler
{
/* ************************************************************************************************
 * Class Constructor
 * ************************************************************************************************
 */ 
public:
    Scheduler_Baseline (CPU* cpu);

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    virtual void Sched () override;

protected:
    virtual bool Inference_Admission ();
    virtual bool Memory_Allocator    ();
    virtual bool Inference_Launcher  ();
};


/** ===============================================================================================
 * \name    Scheduler_Average
 * 
 * \brief   ...
 * 
 * \endcond
 * ================================================================================================
 */
class Scheduler_Average : public Scheduler_Baseline
{
/* ************************************************************************************************
 * Class Constructor
 * ************************************************************************************************
 */ 
public:
    Scheduler_Average (CPU* cpu);

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
protected:
    bool Inference_Admission () override;
};



/** ===============================================================================================
 * \name    Scheduler_BARM
 * 
 * \brief   ...
 * 
 * \endcond
 * ================================================================================================
 */
class Scheduler_BARM : public Scheduler_Baseline
{
/* ************************************************************************************************
 * Class Constructor
 * ************************************************************************************************
 */ 
public:
    Scheduler_BARM (CPU* cpu);

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    void Sched () override;

private:
    bool BASMD  ();
    bool TPMEMA ();
};



/** ===============================================================================================
 * \name    Scheduler_SALBI
 * 
 * \brief   ...
 * 
 * \endcond
 * ================================================================================================
 */
class Scheduler_SALBI : public Scheduler
{
/* ************************************************************************************************
 * Class Constructor
 * ************************************************************************************************
 */ 
public:
    Scheduler_SALBI (CPU* cpu);

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    void Sched () override;

private:
    bool WASMD ();
    bool ORBIS ();
    bool BCLA  ();
};

#endif