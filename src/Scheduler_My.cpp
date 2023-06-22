/**
 * \name    Scheduler_My.cpp
 * 
 * \brief   Implement my function used in CPU.hpp.
 * 
 * \details ...
 * 
 * \date    Jun 22, 2023
 */
#include "include/Scheduler.hpp"

map<int, list<Model*>> abandonedModels = {};

bool APP_Level_SM_Allocator   (CPU* mCPU);
bool Model_Level_SM_Allocator (CPU* mCPU);

/** ===============================================================================================
 * \name    Inference_Admission_API::My
 * 
 * \brief   ...
 * 
 * \note    cannot terminate model when it's kernel is already running due to the sharing filter
 * 
 * \endcond
 * ================================================================================================
 */
bool 
Inference_Admission_API::My (CPU* mCPU)
{  
    log_T("CPU", "Inference_Admission: My");

    for (auto& models :  abandonedModels)
    {
        auto model_info = mCPU->mAPPs[models.first]->modelInfo;
        for (auto model = models.second.begin(); model != models.second.end();)
        {
            if (((*model)->task.deadLine - model_info.totalExecuteTime) <= total_gpu_cycle) models.second.erase(model++);
            else model++;
        }
    }

    /* *******************************************************************
     * Two level SM allocation
     * *******************************************************************
     */
    APP_Level_SM_Allocator (mCPU);

    return Model_Level_SM_Allocator (mCPU);
}

/** ===============================================================================================
 * \name    APP_Level_SM_Allocator
 * 
 * \brief   Allocate SM to applications by the total model workload
 * 
 * \endcond
 * ================================================================================================
 */
bool APP_Level_SM_Allocator (CPU* mCPU)
{
    /* *******************************************************************
     * Reset all SM allocation
     * *******************************************************************
     */
    for (auto app : mCPU->mAPPs) app->SM_budget = {};

    /* *******************************************************************
     * Record needed informations
     * *******************************************************************
     */
    int total_workload = 0;
    list<pair<int, Application*>> app_list;
    for (auto app : mCPU->mAPPs) 
    {
        int model_count = app->runningModels.size() + app->waitingModels.size();
        if (model_count)
        {
            float workload = app->modelInfo.numOfRequest * model_count * app->modelInfo.ioMemCount / app->modelInfo.filterMemCount;
            total_workload += workload;
            app_list.emplace_back(make_pair(workload, app));
        }
    }
    if (app_list.empty()) return false;

    /* sort the applications in non-decreasing workload order */
    app_list.sort([](const auto& a, const auto& b){return a.first < b.first;});

    /* *******************************************************************
     * Allocate SM to application
     * *******************************************************************
     */
    int sm_count = 0, sm_budget = system_resource.SM_NUM;

    /* starvation avoidance */
    while (1)
    {
        auto app = app_list.front();
        if (round((float) sm_budget * app.first / total_workload) == 0)
        {
            total_workload -= app.first;
            app.second->SM_budget.insert(sm_count++);
            sm_budget--;
            
            app_list.pop_front();
        } 
        else break;
    }

    /* allocation by workload ratio */
    for (auto app : app_list)
    {
        int sm_num = round((float) sm_budget * app.first / total_workload);
        
        for (int i = 0; i < sm_num; i++) 
        {
            app.second->SM_budget.insert(sm_count++);
            if (sm_count == system_resource.SM_NUM) break;
        }
    }
    /* mend up the round to zero issue which remain one SM not allocated */
    if (sm_count < system_resource.SM_NUM) app_list.front().second->SM_budget.insert(sm_count++);

#if (PRINT_SM_ALLCOATION_RESULT)
    for (auto app : mCPU->mAPPs) 
    {
        std::cout << "App" << app->appID << ": ";
        for (auto sm_id : app->SM_budget) std::cout << sm_id << ", ";
        std::cout << std::endl;
    }
#endif

    return true;
}


/** ===============================================================================================
 * \name    Model_Level_SM_Allocator
 * 
 * \brief   Allocate the SM to the existing models
 * 
 * \endcond
 * ================================================================================================
 */
bool Model_Level_SM_Allocator (CPU* mCPU)
{
    bool new_model_admission = false;
    for (auto app : mCPU->mAPPs)
    {
        if (app->SM_budget.empty()) continue;

        unordered_set<int> used_sm = {}, available_sm = app->SM_budget;
        for (auto model : app->runningModels) used_sm.insert(model->SM_budget.begin(), model->SM_budget.end());

        for (auto sm_id : used_sm) if (app->SM_budget.count(sm_id) > 0) available_sm.erase(sm_id);

        /* no extra sm allocated to this application */
        if (available_sm.empty()) continue;

        /* Launch new model to inference from waiting queue */
        if (!app->waitingModels.empty())
        {
#if (ENABLE_DEADLINE)
            double BBR          = (double) app->modelInfo.ioMemCount / (app->modelInfo.ioMemCount + app->modelInfo.filterMemCount);
            double sm_ratio     = (double) available_sm.size() / system_resource.SM_NUM;
            double num_of_model = (double) (app->waitingModels.front()->task.deadLine - total_gpu_cycle) / app->modelInfo.totalExecuteTime;
            int batch_limit     = min((int)floor(sm_ratio * num_of_model / BBR), (int)app->waitingModels.size());
#else
            int batch_limit     = (double) app->waitingModels.size();
#endif
            std::cout << "App " << app->appID << " has " << batch_limit << " batch limit" << std::endl;

            for (int i = 0; i < batch_limit; i++)
            {
                auto model = app->waitingModels.front();
                model->SM_budget = available_sm;

#if (PRINT_SM_ALLCOATION_RESULT)
                std::cout << "APP: " << app->appID << " Model: " << model->modelID << " get SM: ";
                for (auto sm_id : model->SM_budget) std::cout << sm_id << ", ";
                std::cout << std::endl;
#endif
                app->runningModels.push_back(model);
                app->waitingModels.pop_front();
            }
        } 
        
        /* Issue the sm to running models */
        else if (!app->runningModels.empty())
        {
            for (auto model : app->runningModels) model->SM_budget.insert(available_sm.begin(), available_sm.end());
        }

        else ASSERT(false, "Allocate SM to application with empty models");

        new_model_admission = true;
    }

    return new_model_admission;
}