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

/** ===============================================================================================
 * \name    Scheduler_SALBI
 * 
 * \brief   ...
 * 
 * \param   cpu     the pointer of CPU
 * 
 * \endcond
 * ================================================================================================
 */ 
Scheduler_SALBI::Scheduler_SALBI (CPU* cpu) : Scheduler(cpu)
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
Scheduler_SALBI::Sched ()
{
    WASMD ();

    ORBIS ();
    
    // BCLA  ();
}


/** ===============================================================================================
 * \name    WASMD : Workload Aware SM Dispatchor
 * 
 * \brief   ...
 * 
 * \endcond
 * ================================================================================================
 */
bool 
Scheduler_SALBI::WASMD ()
{
    log_T("Scheduler_SALBI", "WASMD");

    for (auto app : mCPU->mAPPs) app->runningModels.splice(app->runningModels.end(), app->waitingModels);

    /* *******************************************************************
     * Record needed informations
     * *******************************************************************
     */
    for (auto app : mCPU->mAPPs) app->SM_budget = {};
    
    list<pair<int, unsigned long long>> workload_list;
    for (auto app : mCPU->mAPPs)
    {
        if (!app->runningModels.empty())
        {
            unsigned long long NP = app->modelInfo.ioMemCount * app->runningModels.size() + app->modelInfo.filterMemCount;
            double BBR = (double)app->modelInfo.filterMemCount / (double)(app->modelInfo.ioMemCount + app->modelInfo.filterMemCount);

            workload_list.emplace_back(make_pair(app->appID, NP * BBR));
        }
    }
    if (workload_list.empty()) return false;

    /* sort the applications in non-decreasing workload order */
    workload_list.sort([](const auto& a, const auto& b){return a.second < b.second;});

    /* *******************************************************************
     * Allocate SM to applications
     * *******************************************************************
     */
    unsigned long long total_workload = 0;
    for (auto app_pair : workload_list) total_workload += app_pair.second;

    int sm_count = 0, sm_budget = system_resource.SM_NUM;
    for (auto app_pair : workload_list)
    {
        int sm_num = max(1, (int)round(sm_budget * (double)app_pair.second / (double)total_workload));
        
        for (int i = 0; i < sm_num; i++) 
        {
            if (sm_count == system_resource.SM_NUM) break;
            mCPU->mAPPs[app_pair.first]->SM_budget.insert(sm_count++);
        }
    }
    /* mend up the round to zero issue which remain one SM not allocated */
    if (sm_count < system_resource.SM_NUM) mCPU->mAPPs.front()->SM_budget.insert(sm_count++);

    return true;
}


/** ===============================================================================================
 * \name    ORBIS : Overhead Reduction Batch Inference Scheduler
 * 
 * \brief   ...
 * 
 * \endcond
 * ================================================================================================
 */
bool
Scheduler_SALBI::ORBIS ()
{
    log_T("Scheduler_SALBI", "ORBIS");

    mCPU->mGPU->getGMMU()->setCGroupType (true);

    unsigned long long memory_budget = system_resource.VRAM_SPACE;

    /* *******************************************************************
     * Collect informations of apps
     * *******************************************************************
     */
    /* Record memory usage */
    map<int, unsigned long long> NP_list;
    for (auto kernel : mCPU->mGPU->runningKernels) NP_list[kernel->appID] += kernel->getKernelInfo().numOfMemory;
    for (auto kernel : mCPU->mGPU->commandQueue)   NP_list[kernel->appID] += kernel->getKernelInfo().numOfMemory;

    /* Record allocated memory */
    map<int, unsigned long long> NPA_list;
    for (auto app_pair : NP_list) NPA_list[app_pair.first] += mCPU->mGPU->getGMMU()->getCGroup(app_pair.first)->size() * PAGE_SIZE;

    /* Calculate remaining memory */
    for (auto app_pair : NPA_list) memory_budget -= app_pair.second;

    /* Get the ready kernels of non-executing applications */
    map<int, list<Kernel*>> ready_kerenls;
    for (auto app : mCPU->mAPPs)
    {
        /* Application has layer executing */
        if (NP_list.find(app->appID) != NP_list.end()) continue;

        /* Collect the ready kernels */
        for (auto model : app->runningModels) ready_kerenls[app->appID].emplace_back(model->findReadyKernels().front());
        if (ready_kerenls[app->appID].empty()) continue;

        /* Sort in non-decreasing order */
        ready_kerenls[app->appID].sort([](Kernel*& a, Kernel*& b){ return a->srcLayer->layerID < b->srcLayer->layerID; });

        /* Pop out the kernels with non smallest layerID */
        auto kernel = ready_kerenls[app->appID].front();
        ready_kerenls[app->appID].remove_if([kernel](Kernel*& k){ return k->srcLayer->layerID > kernel->srcLayer->layerID; });

        /* Record memory usage */
        NP_list[app->appID] += kernel->srcLayer->getFilterMemory();
        NP_list[app->appID] += (kernel->srcLayer->getIFMapMemory() + kernel->srcLayer->getOFMapMemory()) * ready_kerenls[app->appID].size();
    }

    /* Calculates page fault ratio */
    vector<pair<int, double>> PFR_list;
    for (auto app_pair : NP_list) PFR_list.emplace_back(make_pair(app_pair.first, NP_list[app_pair.first] * (double)(NP_list[app_pair.first] - NPA_list[app_pair.first] + 1) / mCPU->mAPPs[app_pair.first]->SM_budget.size()));
    
    /* Sort in non-decreasing order, the smaller PFR has higher allocation priority */
    sort(PFR_list.begin(), PFR_list.end(), [](auto& a, auto& b){ return a.second < b.second; });


    /* *******************************************************************
     * Allocate memory to applications
     * *******************************************************************
     */
    for (auto app_pair : PFR_list)
    {
        /* This layer of application is already executing with memory oversubscription */
        if (NPA_list[app_pair.first])
        {
            unsigned long long NP_diff = NP_list[app_pair.first] - NPA_list[app_pair.first];

        unsigned long long new_allocate = (NP_diff <= memory_budget) ? NP_diff : memory_budget;

            NPA_list[app_pair.first] += new_allocate;
        
        memory_budget -= new_allocate;
        }
        /* This layer of application is going to excute without memory oversubscription */
        else if (NP_list[app_pair.first] <= memory_budget)
        {
            NPA_list[app_pair.first] = NP_list[app_pair.first];

            memory_budget -= NPA_list[app_pair.first];
        }
        /* This layer of application is going to excute with memory oversubscription */
        else
        {
            NPA_list[app_pair.first] += memory_budget;

            memory_budget  = 0;
        }
    }
    ASSERT(memory_budget <= system_resource.VRAM_SPACE, "Allocation overflow");


    /* Perform the GPU memory isolation setup for applications */
    for (auto app_pair : NPA_list)
    {
        mCPU->mGPU->getGMMU()->setCGroupSize(app_pair.first, app_pair.second / PAGE_SIZE);
    }

    /* *******************************************************************
     * Record the blocking SM
     * *******************************************************************
     */
    unordered_set<int> blocking_SMs;
    for (auto app_pair : ready_kerenls)
    {
        if (NPA_list[app_pair.first] == 0)
        {
            blocking_SMs.insert(mCPU->mAPPs[app_pair.first]->SM_budget.begin(), mCPU->mAPPs[app_pair.first]->SM_budget.end());
        }
    }

    for (auto app_pair : PFR_list)
    {
        if (ready_kerenls.find(app_pair.first) != ready_kerenls.end())
        {
            app_pair.second = NP_list[app_pair.first] * (double)(NP_list[app_pair.first] - NPA_list[app_pair.first] + 1) / (double)(mCPU->mAPPs[app_pair.first]->SM_budget.size() + blocking_SMs.size());
        }
    }
    sort(PFR_list.begin(), PFR_list.end(), [](auto& a, auto& b){ return a.second < b.second; });

    /* *******************************************************************
     * Batch inference the layer according to the allocated memory space
     * *******************************************************************
     */
    for (auto app_pair : PFR_list)
    {
        if (ready_kerenls.find(app_pair.first) == ready_kerenls.end() ||  NPA_list[app_pair.first] == 0) continue;

        auto available_sm = mCPU->mGPU->getIdleSMs();
        for (auto sm_id : mCPU->mAPPs[app_pair.first]->SM_budget) if (!available_sm.count(sm_id)) continue;

        auto kernel_list = ready_kerenls[app_pair.first];

        int batch_size = max(1, (int)ceil((double)(NPA_list[app_pair.first] - kernel_list.front()->srcLayer->getFilterMemory()) / (double)(kernel_list.front()->srcLayer->getIFMapMemory() + kernel_list.front()->srcLayer->getOFMapMemory())));

        if (strcmp(kernel_list.front()->srcLayer->layerType, "Dense") == 0) batch_size = kernel_list.size();

        vector<pair<Kernel*, int>> sync_kernels;
        for (auto k : kernel_list)
        {
            if (sync_kernels.size() < batch_size)
            {
                sync_kernels.push_back(make_pair(k, 1));
            }
        }

        Kernel* kernel  = new KernelGroup(sync_kernels);
        kernel->SM_List = new unordered_set<int> (mCPU->mAPPs[app_pair.first]->SM_budget);
        
        if (!blocking_SMs.empty() && app_pair == PFR_list.front())
        {
            log("BASIA", "app " + to_string(app_pair.first) + " lending " + to_string(blocking_SMs.size()) + " SMs", Color::Yellow);
            std::cout << "From " << kernel->SM_List->size() << " to ";
            kernel->SM_List->insert(blocking_SMs.begin(), blocking_SMs.end());
            std::cout << kernel->SM_List->size() << std:: endl;
        }
        blocking_SMs.clear();        

        kernelLauncher(move(kernel));
    }

    return true;
}


/** ===============================================================================================
 * \name    BCLA : Blocking Core Lending Algorithm
 * 
 * \brief   ...
 * 
 * \endcond
 * ================================================================================================
 */
bool
Scheduler_SALBI::BCLA ()
{
    return true;
}