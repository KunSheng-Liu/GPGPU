/**
 * \name    Scheduler_Average.cpp
 * 
 * \brief   ...
 * 
 * \date    Jul 22, 2023
 */
#include "include/Scheduler.hpp"

/** ===============================================================================================
 * \name    Scheduler_Average
 * 
 * \brief   ...
 * 
 * \param   cpu     the pointer of CPU
 * 
 * \endcond
 * ================================================================================================
 */ 
Scheduler_Average::Scheduler_Average (CPU* cpu) : Scheduler_Baseline(cpu)
{

}


/** ===============================================================================================
 * \name    Inference_Admission
 * 
 * \brief   Assign all SM to every applications
 * 
 * \note    * This defualt scheduler didn't block SM to any application, which leads the kernel contension
 *          the SM resource in GPU core.
 * 
 * \endcond
 * ================================================================================================
 */
bool 
Scheduler_Average::Inference_Admission ()
{
    int sm_count = 0, sm_budget = system_resource.SM_NUM;
    while (sm_budget != 0) 
    {
        for (auto app : mCPU->mAPPs) 
        {
            app->SM_budget.insert(sm_count++);
            if (!(--sm_budget)) break;
        }
    }

    for (auto app : mCPU->mAPPs) app->runningModels.splice(app->runningModels.end(), app->waitingModels);

    return true;
}