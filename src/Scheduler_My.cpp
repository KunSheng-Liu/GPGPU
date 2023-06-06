/**
 * \name    Approach_My.cpp
 * 
 * \brief   Implement my function  used in CPU.hpp.
 * 
 * \details ...
 * 
 * \date    Jun 3, 2023
 */
#include "include/Scheduler.hpp"


/** ===============================================================================================
 * \name    LazyB_Inference_Admission
 * 
 * \brief   Only one model can be running in a time, therefore launch the tasks that can be merge
 *          into the model without violate the deadline.
 * 
 * \note    cannot terminate model when it's kernel is already running due to the sharing filter
 * 
 * \endcond
 * ================================================================================================
 */
bool 
Scheduler_My::Inference_Admission ()
{  
    log_T("CPU", "Inference_Admission: My");

    /* *******************************************************************
     * Allocate SM to application
     * *******************************************************************
     */
    int total_req = 0;
    for (auto app : mCPU->mAPPs) total_req += app->modelInfo.numOfRequest * app->runningModels.size();
    if (total_req == 0) return false;

    list<pair<int, Application*>> app_list;
    for (auto app : mCPU->mAPPs) 
    {
        app_list.push_back(make_pair(app->modelInfo.numOfRequest * app->runningModels.size(), app));
    }

    app_list.sort([](const auto& a, const auto& b){return a.first < b.first;});

    int sm_count = 0, sm_budget = GPU_SM_NUM;
    for (auto app : app_list)
    {
        app.second->SM_budget = {};

        if (app.first)
        {
            int sm_num = round((float) sm_budget * app.first / total_req);
            for (int i = 0; i < sm_num; i++) 
            {
                app.second->SM_budget.insert(sm_count++);
                if (sm_count == GPU_SM_NUM) break;
            }

            if (app.second->SM_budget.empty())
            {
                total_req -= app.first;
                app.second->SM_budget.insert(sm_count++);
                sm_budget--;
            }
        }        
    }
    /* mend up the round to zero issue which remain one SM not allocated */
    if (sm_count < GPU_SM_NUM) mCPU->mAPPs.front()->SM_budget.insert(sm_count++);

    /* *******************************************************************
     * ration SM to models
     * *******************************************************************
     */
    for (auto app : mCPU->mAPPs)
    {
        cout << "App" << app->appID << ": ";
        for (auto sm_id : app->SM_budget) cout << sm_id << ", ";
        cout << endl;

        for (auto model : app->runningModels) model->SM_budget = app->SM_budget;
    }

    return true;
}


/** ===============================================================================================
 * \name    LazyB_Kernel_Scheduler
 * 
 * \brief   Launch the smallest ready kernel of all the models, if multiple model has smallest ready 
 *          kernels merge the models.
 * 
 * \endcond
 * ================================================================================================
 */
bool 
Scheduler_My::Kernel_Scheduler ()
{  
    log_T("CPU", "Kernel_Scheduler: My");

    Scheduler::Kernel_Scheduler();

    return false;
}