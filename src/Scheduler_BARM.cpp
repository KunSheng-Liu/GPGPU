/**
 * \name    Approach_BARM.cpp
 * 
 * \brief   Implement the callback function for related work \b BARM used in CPU.hpp.
 * 
 * \date    May 25, 2023
 */
#include "include/Scheduler.hpp"


/** ===============================================================================================
 * \name    BARM_Inference_Admission
 * 
 * \brief   Use BARM::SMD scheme to allocate SM
 * 
 * \endcond
 * ================================================================================================
 */
bool 
Scheduler_BARM::Inference_Admission ()
{  
    log_T("CPU", "Inference_Admission: BARM");

    /*  Get avaiable SM list */
    list<int> available_sm = mCPU->mGPU->getIdleSMs();
    if (available_sm.empty()) return false;

    ASSERT(false, "haven't implement SMD");

    /* *******************************************************************
     * Assign SM to each application according to its model needed pages
     * *******************************************************************
     */
    /* Record the total required memory base on the task number */
    // float total_needed_memory = 0;
    // vector<pair<float, Application*>> APP_list;
    // for (auto app : mCPU->mAPPs)
    // {
    //     app->SM_budget = {};

    //     if(app->tasks.size() == 0) continue;

    //     auto info = app->modelInfo;

    //     total_needed_memory += (info.filterMemCount + info.filterMemCount) * app->tasks.size();  
        
    //     APP_list.emplace_back(make_pair((info.filterMemCount + info.filterMemCount) * app->tasks.size(), app));     
    // }

    // /* Sort to non-decreacing order */
    // sort(APP_list.begin(), APP_list.end(), [](const pair<float, Application*>& a, const pair<float, Application*>& b){
    //     return a.first < b.first;
    // });

    // /* Assign SM to each application */
    // int SM_count = 0;
    // for (auto app_pair : APP_list)
    // {
    //     std::cout << GPU_SM_NUM * (app_pair.first / total_needed_memory) << std::endl;
    //     /* Avoid starvation, at least assign 1 SM to application */
    //     if ((int)(GPU_SM_NUM * (app_pair.first / total_needed_memory) == 0))
    //     {
    //         total_needed_memory -= app_pair.first;
    //         app_pair.second->SM_budget.push_back(SM_count++);
    //         continue;
    //     }

    //     for (int i = 0; i < (int)(GPU_SM_NUM * (app_pair.first / total_needed_memory)); i++)
    //     {
    //         app_pair.second->SM_budget.push_back(SM_count++);
    //     }

    //     ASSERT(SM_count == GPU_SM_NUM);
    // }

}
