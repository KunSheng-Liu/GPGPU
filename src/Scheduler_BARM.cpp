/**
 * \name    Scheduler_BARM.cpp
 * 
 * \brief   Implement the callback function for related work \b BARM used in CPU.hpp.
 * 
 * \date    Jun 22, 2023
 */
#include "include/Scheduler.hpp"


/** ===============================================================================================
 * \name    Inference_Admission_API::BARM
 * 
 * \brief   Use BARM::SMD scheme to allocate SM
 * 
 * \endcond
 * ================================================================================================
 */
bool 
Inference_Admission_API::BARM (CPU* mCPU)
{  
    log_T("CPU", "Inference_Admission: BARM");

    /* *******************************************************************
     * Assign SM to each application according to its model needed pages
     * *******************************************************************
     */
    /* Record the total required memory base on the task number */
    unsigned long long NP_total = 0;
    list<pair<int, unsigned long long>> NP_list;
    for (auto app : mCPU->mAPPs)
    {
        unsigned long long NP = app->modelInfo.ioMemCount * app->runningModels.size() + app->modelInfo.filterMemCount;

        NP_list.emplace_back(make_pair(app->appID, NP));
        NP_total += NP;
    }
    if (NP_list.empty()) return false;

    /* Sort to non-decreacing order */
    NP_list.sort([](const auto& a, const auto& b){return a.second < b.second;});

    /* Assign SM to each application */
    int sm_count = 0, sm_budget = system_resource.SM_NUM;

    for (auto app_pair : NP_list)
    {
        int sm_num = round((float) sm_budget * app_pair.second / NP_total);
        
        for (int i = 0; i < sm_num; i++) 
        {
            mCPU->mAPPs[app_pair.first]->SM_budget.insert(sm_count++);
            if (sm_count == system_resource.SM_NUM) break;
        }
    }
    /* mend up the round to zero issue which remain one SM not allocated */
    if (sm_count < system_resource.SM_NUM) mCPU->mAPPs.front()->SM_budget.insert(sm_count++);

        /* *******************************************************************
     * Assign SM to each application
     * *******************************************************************
     */
    for (auto app : mCPU->mAPPs)
    {
        while(!app->waitingModels.empty())
        {
            auto model = app->waitingModels.front();
            model->SM_budget = unordered_set<int> (app->SM_budget);

#if (PRINT_SM_ALLCOATION_RESULT)
            std::cout << "APP: " << app->appID << " Model: " << model->modelID << " get SM: ";
            for (auto sm_id : model->SM_budget) std::cout << sm_id << ", ";
            std::cout << std::endl;
#endif
            app->runningModels.push_back(model);
            app->waitingModels.pop_front();
        }
    }

    return true;
}


/** ===============================================================================================
 * \name    MEMA
 * 
 * \brief   Use BARM::MEMA scheme to allocate memory
 * 
 * \endcond
 * ================================================================================================
 */
bool 
Memory_Allocator_API::MEMA (CPU* mCPU)
{
    map<int, unsigned long long> memory_record;

    for (auto kernel : mCPU->mGPU->runningKernels) memory_record[kernel->appID] += ceil(kernel->getKernelInfo().numOfMemory / PAGE_SIZE);
    for (auto kernel : mCPU->mGPU->commandQueue)   memory_record[kernel->appID] += ceil(kernel->getKernelInfo().numOfMemory / PAGE_SIZE);

    vector<pair<int, unsigned long long>> memory_budget = vector<pair<int, unsigned long long>> (memory_record.begin(), memory_record.end());
    sort(memory_budget.begin(), memory_budget.end(), [](auto& a, auto& b){ return a.second < b.second; });

    int app_num = memory_budget.size();
    unsigned long long remaining_pages = system_resource.VRAM_SPACE / PAGE_SIZE;
    for (auto& memory_pair : memory_budget)
    {
        if (remaining_pages < memory_pair.second) memory_pair.second = floor(remaining_pages / app_num);

        remaining_pages -= memory_pair.second;
        app_num--;
    }

    unsigned long long extra_memory = floor(remaining_pages / memory_budget.size());
    for (auto& memory_pair : memory_budget)
    {
        memory_pair.second += extra_memory;
        remaining_pages    -= extra_memory;
    }

    for (auto& memory_pair : memory_budget)
    {
        if (remaining_pages-- == 0) break;
        memory_pair.second++;
    }

    for (auto& memory_pair : memory_budget) mCPU->mGPU->getGMMU()->setCGroupSize(memory_pair.first, memory_pair.second);
    
    return true;
}


/** ===============================================================================================
 * \name    R_MEMA
 * 
 * \brief   Use BARM::R_MEMA scheme to allocate memory
 * 
 * \endcond
 * ================================================================================================
 */
bool 
Memory_Allocator_API::R_MEMA (CPU* mCPU)
{  
    return true;
}