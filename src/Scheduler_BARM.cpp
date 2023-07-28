/**
 * \name    Scheduler_BARM.cpp
 * 
 * \brief   Implement the callback function for related work \b BARM used in CPU.hpp.
 * 
 * \date    Jun 22, 2023
 */
#include "include/Scheduler.hpp"

/** ===============================================================================================
 * \name    Scheduler_BARM
 * 
 * \brief   ...
 * 
 * \param   cpu     the pointer of CPU
 * 
 * \endcond
 * ================================================================================================
 */ 
Scheduler_BARM::Scheduler_BARM (CPU* cpu) : Scheduler_Baseline(cpu)
{

}


/** ===============================================================================================
 * \name    Sched
 * 
 * \brief   perform schedule
 * 
 * \endcond
 * ================================================================================================
 */
void 
Scheduler_BARM::Sched ()
{
    BASMD();

    Inference_Launcher();

    TPMEMA();
}


/** ===============================================================================================
 * \name    BASMD
 * 
 * \brief   Use BARM::SMD scheme to allocate SM
 * 
 * \endcond
 * ================================================================================================
 */
bool 
Scheduler_BARM::BASMD ()
{
    log_T("CPU", "Inference_Admission: BARM");

    for (auto app : mCPU->mAPPs) app->runningModels.splice(app->runningModels.end(), app->waitingModels);

    /* *******************************************************************
     * Record needed informations
     * *******************************************************************
     */
    list<pair<int, unsigned long long>> NP_list;
    for (auto app : mCPU->mAPPs) app->SM_budget = {};
    for (auto app : mCPU->mAPPs) NP_list.emplace_back(make_pair(app->appID, app->modelInfo.ioMemCount * app->runningModels.size() + app->modelInfo.filterMemCount));

    if (NP_list.empty()) return false;

    /* Sort to non-decreacing order */
    NP_list.sort([](const auto& a, const auto& b){return a.second < b.second;});

    /* *******************************************************************
     * Allocate SM to applications
     * *******************************************************************
     */
    unsigned long long total_NP = 0;
    for (auto app_pair : NP_list) total_NP += app_pair.second;

    int sm_count = 0, sm_budget = system_resource.SM_NUM;
    for (auto app_pair : NP_list)
    {
        int sm_num = max(1, (int)round(sm_budget * (double)app_pair.second / (double)total_NP));
        
        for (int i = 0; i < sm_num; i++) 
        {
            mCPU->mAPPs[app_pair.first]->SM_budget.insert(sm_count++);
            if (sm_count == system_resource.SM_NUM) break;
        }
    }
    /* mend up the round to zero issue which remain one SM not allocated */
    if (sm_count < system_resource.SM_NUM) mCPU->mAPPs.front()->SM_budget.insert(sm_count++);

    return true;
}


/** ===============================================================================================
 * \name    TPMEMA
 * 
 * \brief   Use BARM::MEMA scheme to allocate memory
 * 
 * \endcond
 * ================================================================================================
 */
bool 
Scheduler_BARM::TPMEMA ()
{
    mCPU->mGPU->getGMMU()->setCGroupType (true);

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