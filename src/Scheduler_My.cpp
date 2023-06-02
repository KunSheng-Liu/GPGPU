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
    int total_Req = 0;
    list<pair<int, Application*>> app_list;
    for (auto app : mCPU->mAPPs) 
    {
        if (!app->runningModels.empty())
        {
            int requestCount = app->modelInfo.numOfRequest * app->runningModels.size();
            total_Req += requestCount;
            app_list.push_back(make_pair(requestCount, app));
        }
    }
    if (!total_Req) return false;
    app_list.sort([](const auto& a, const auto& b){return a.first < b.first;});

    int SM_count = 0, SM_budge = GPU_SM_NUM;
    for (auto app : app_list)
    {
        app.second->SM_budget = {};
        int sm_num = round((float) SM_budge * app.first / total_Req);
        for (int i = 0; i < sm_num; i++) app.second->SM_budget.insert(SM_count++);

        if (app.second->SM_budget.empty())
        {
            total_Req -= app.first;
            app.second->SM_budget.insert(SM_count++);
            SM_budge--;
        }
        
        cout << "App" << app.second->appID << ": ";
        for (auto sm_id : app.second->SM_budget) cout << sm_id << ", ";
        cout << endl;
    }

    /* *******************************************************************
     * ration SM to models
     * *******************************************************************
     */
    for (auto app : mCPU->mAPPs)
    {
        for (auto model : app->runningModels)
        {
            model->SM_budget = app->SM_budget;
        }
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