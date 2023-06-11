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
 * \name    Scheduler | Scheduler_Baseline
 * 
 * \brief   The base class of NN scheduler. You can add new schedule policy by inheritance this class 
 *          and override the virtual function to fit the desire.
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
public:
    Scheduler (CPU* cpu) : mCPU(cpu) {}

public:
    virtual bool Inference_Admission ();
    virtual bool Kernel_Scheduler    ();
    virtual bool Memory_Allocator    ();

protected:
    void missDeadlineHandler ();

protected:
    CPU* mCPU;
};
typedef Scheduler Scheduler_Baseline;



/** ===============================================================================================
 * \name    Scheduler_Greedy
 * 
 * \brief   A greedly NN inference scheduler that inherits the \b "Scheduler" class. 
 * 
 * \note    * This scheduler blocks SM to an application, no model can be build until the running 
 *          model is totally finished.
 * 
 * \endcond
 * ================================================================================================
 */
class Scheduler_Greedy : public Scheduler
{
public:
    Scheduler_Greedy (CPU* cpu) : Scheduler(cpu) {}

public:
    bool Inference_Admission () override;
};



/** ===============================================================================================
 * \name    Scheduler_BARM
 * 
 * \brief   Related work NN inference scheduler \b "BARM" that inherits the \b "Scheduler" class. 
 * 
 * \endcond
 * ================================================================================================
 */
class Scheduler_BARM : public Scheduler
{
public:
    Scheduler_BARM (CPU* cpu) : Scheduler(cpu) {}

public:
    bool Inference_Admission () override;
};




/** ===============================================================================================
 * \name    Scheduler_LazyB
 * 
 * \brief   Related work NN inference scheduler \b "Lazy_Batching" that inherits the \b "Scheduler" class. 
 * 
 * \details As a cloud server, the tasks is launched form the edge devices. Therefore try to maximize 
 *          the batch size of the task in kernel level.
 * 
 * \note    Use resnet model
 * \note    In this scenario, no memory limitation to the system.
 * \note    This approach use ResNet model
 * \note    Max batch size is constrained as 64
 * \note    The memory access overhead is set as 100 cycle
 * 
 * \endcond
 * ================================================================================================
 */
class Scheduler_LazyB : public Scheduler
{
    #define LAZYB_MAX_CONCURRENCY           GPU_SM_NUM 
    #define LAZYB_MAX_BATCH_SIZE            64 
    #define LAZYB_CYCLE_DEADLINE            GPU_F / 10
    
public:
    Scheduler_LazyB (CPU* cpu) : Scheduler(cpu) {}

public:
    bool Inference_Admission () override;
    bool Kernel_Scheduler    () override;

};


/* ************************************************************************************************
 * My
 * ************************************************************************************************
 */
class Scheduler_My : public Scheduler
{
public:
    Scheduler_My (CPU* cpu) : Scheduler(cpu) {}

public:
    bool Inference_Admission () override;
    bool Kernel_Scheduler    () override;

protected:
    bool Workload_SM_Allocator ();
    bool Rescue_SM_Allocator   ();
    
    bool APP_Level_SM_Allocator ();
    bool Model_Level_SM_Allocator ();

private:
    map<int, list<Model*>> abandonedModels = {};

};

#endif