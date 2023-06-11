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
 * \name    My_Inference_Admission
 * 
 * \brief   ...
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
     * Check the models haven't miss deadline, if so, terminate model
     * *******************************************************************
     */
    missDeadlineHandler ();

    for (auto& models :  abandonedModels)
    {
        auto model_info = mCPU->mAPPs[models.first]->modelInfo;
        for (auto model = models.second.begin(); model != models.second.end();)
        {
            if (((*model)->task.deadLine - model_info.totalExecuteTime) <= total_gpu_cycle) models.second.erase(model++);
            else model++;
        }
    }

    // /* *******************************************************************
    //  * Check new model arrival
    //  * *******************************************************************
    //  */
    // bool new_arrival = false;
    // for (auto app : mCPU->mAPPs) new_arrival |= !app->waitingModels.empty();

    // if (new_arrival) return Workload_SM_Allocator();

    // /* *******************************************************************
    //  * Check some model is be abandoned
    //  * *******************************************************************
    //  */
    // bool rescue_model = false;
    // for (auto app : mCPU->mAPPs) rescue_model |= !abandonedModels[app->appID].empty();

    // return rescue_model ? Rescue_SM_Allocator() : Workload_SM_Allocator();

    /* *******************************************************************
     * Two level SM allocation
     * *******************************************************************
     */
    APP_Level_SM_Allocator();

    return Model_Level_SM_Allocator();
}


/** ===============================================================================================
 * \name    My_Kernel_Scheduler
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


/** ===============================================================================================
 * \name    Workload_SM_Allocator
 * 
 * \brief   Allocate the SM according to the workload ratio
 * 
 * \endcond
 * ================================================================================================
 */
bool 
Scheduler_My::Workload_SM_Allocator ()
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
    int sm_count = 0, sm_budget = GPU_SM_NUM;

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
            if (sm_count == GPU_SM_NUM) break;
        }
    }
    /* mend up the round to zero issue which remain one SM not allocated */
    if (sm_count < GPU_SM_NUM) app_list.front().second->SM_budget.insert(sm_count++);

#if (PRINT_SM_ALLCOATION_RESULT)
    for (auto app : mCPU->mAPPs) 
    {
        cout << "App" << app->appID << ": ";
        for (auto sm_id : app->SM_budget) cout << sm_id << ", ";
        cout << endl;
    }
#endif

    /* *******************************************************************
     * ration SM to models
     * *******************************************************************
     */
    for (auto app : mCPU->mAPPs)
    {
        if (!app->waitingModels.empty())
        {
            double BBR          = (double) app->modelInfo.ioMemCount / app->modelInfo.filterMemCount;
            double sm_ratio     = (double) app->SM_budget.size() / GPU_SM_NUM;
            double num_of_model = (double) (app->waitingModels.front()->task.deadLine - total_gpu_cycle) / app->modelInfo.totalExecuteTime;

            int batch_limit = min((int)floor(sm_ratio * num_of_model / BBR), (int)app->waitingModels.size());
            cout << "App " << app->appID << " has " << (int)(sm_ratio * num_of_model / BBR) << " batch limit" << endl;

            for (int i = 0; i < batch_limit; i++)
            {
                auto model = app->waitingModels.front();
                model->SM_budget = app->SM_budget;

#if (PRINT_SM_ALLCOATION_RESULT)
                std::cout << "APP: " << app->appID << " Model: " << model->modelID << " get SM: ";
                for (auto sm_id : model->SM_budget) std::cout << sm_id << ", ";
                std::cout << std::endl;
#endif
                app->runningModels.push_back(model);
                app->waitingModels.pop_front();
            }
        
            /* Stash the model,  */
            abandonedModels[app->appID].splice(abandonedModels[app->appID].end(), app->waitingModels);
        }
    }
    return true;
}


/** ===============================================================================================
 * \name    Rescue_SM_Allocator
 * 
 * \brief   
 * 
 * \endcond
 * ================================================================================================
 */
bool 
Scheduler_My::Rescue_SM_Allocator ()
{
    /* *******************************************************************
     * Check application finished with SM
     * *******************************************************************
     */
    unordered_set<int> released_sm;
    for (auto app : mCPU->mAPPs) 
    {
        if (app->runningModels.empty() && !app->SM_budget.empty()) released_sm.insert(app->SM_budget.begin(), app->SM_budget.end());
    }
    if (released_sm.empty()) return false;

    int app_id = -1;
    int batch_limit = 0;
    for (auto models : abandonedModels)
    {
        if (!models.second.empty())
        {
            auto   model_info   = mCPU->mAPPs[models.first]->modelInfo;
            double BBR          = (double) model_info.ioMemCount / model_info.filterMemCount;
            double sm_ratio     = (double) released_sm.size() / GPU_SM_NUM;
            double num_of_model = (double) (models.second.front()->task.deadLine - total_gpu_cycle) / model_info.totalExecuteTime;

            int batch_size = min((int)floor(sm_ratio * num_of_model / BBR), (int)models.second.size());

            if (batch_size > batch_limit) 
            {
                batch_limit = batch_size;
                app_id = models.first;
            }
        }
    }
    if (app_id == -1) return false;

    /* *******************************************************************
     * ration SM to models
     * *******************************************************************
     */
    auto model = abandonedModels[app_id].front();
    model->SM_budget = released_sm;

#if (PRINT_SM_ALLCOATION_RESULT)
    auto app = mCPU->mAPPs[app_id];
    std::cout << "APP: " << app->appID << " Model: " << model->modelID << " get SM: ";
    for (auto sm_id : model->SM_budget) std::cout << sm_id << ", ";
    std::cout << std::endl;
#endif
    app->runningModels.push_back(model);
    abandonedModels[app_id].pop_front();

    return true;
}


/** ===============================================================================================
 * \name    APP_Level_SM_Allocator
 * 
 * \brief   Allocate SM to applications by the total model workload
 * 
 * \endcond
 * ================================================================================================
 */
bool 
Scheduler_My::APP_Level_SM_Allocator ()
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
    int sm_count = 0, sm_budget = GPU_SM_NUM;

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
            if (sm_count == GPU_SM_NUM) break;
        }
    }
    /* mend up the round to zero issue which remain one SM not allocated */
    if (sm_count < GPU_SM_NUM) app_list.front().second->SM_budget.insert(sm_count++);

#if (PRINT_SM_ALLCOATION_RESULT)
    for (auto app : mCPU->mAPPs) 
    {
        cout << "App" << app->appID << ": ";
        for (auto sm_id : app->SM_budget) cout << sm_id << ", ";
        cout << endl;
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
bool 
Scheduler_My::Model_Level_SM_Allocator ()
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
            double BBR          = (double) app->modelInfo.ioMemCount / (app->modelInfo.ioMemCount + app->modelInfo.filterMemCount);
            double sm_ratio     = (double) available_sm.size() / GPU_SM_NUM;
            double num_of_model = (double) (app->waitingModels.front()->task.deadLine - total_gpu_cycle) / app->modelInfo.totalExecuteTime;

            int batch_limit = min((int)floor(sm_ratio * num_of_model / BBR), (int)app->waitingModels.size());
            cout << "App " << app->appID << " has " << (int)(sm_ratio * num_of_model / BBR) << " batch limit" << endl;

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