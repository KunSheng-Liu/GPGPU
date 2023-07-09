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
    unsigned long long total_workload = 0;
    list<pair<unsigned long long, Application*>> app_list;
    for (auto app : mCPU->mAPPs) 
    {
        int model_count = app->runningModels.size() + app->waitingModels.size();
        if (model_count)
        {
            long long workload = app->modelInfo.numOfRequest * model_count;
            app_list.emplace_back(make_pair(workload, app));
            total_workload += workload;
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


/** ===============================================================================================
 * \name    Kernel_Scheduler_API::My
 * 
 * \brief   Default launch kernel in max batch size and reduce the batch size of smallest kernels 
 *          to fit into the current system memory
 * 
 * \note    In this scenario, each application can launch at most one KernelGroup
 * 
 * \endcond
 * ================================================================================================
 */
bool
Kernel_Scheduler_API::My (CPU* mCPU)
{  
    log_T("CPU", "Kernel_Scheduler: My");

    map<int, unsigned long long> memory_record;
    for (auto kernel : mCPU->mGPU->runningKernels) memory_record[kernel->appID] += kernel->getKernelInfo().numOfMemory;

    map<int, list<Kernel*>> kerenls_table;
    for (auto app : mCPU->mAPPs)
    {
        /* application has launched kernel */
        if (memory_record[app->appID]) continue;

        /* *******************************************************************
         * Collect the ready kernels
         * *******************************************************************
         */
        list<Kernel*> readyKernels;
        for (auto model : app->runningModels)
        {
            for (auto kernel : model->findReadyKernels())
            {
                kernel->SM_List = &model->SM_budget;
                readyKernels.push_back(kernel);
            }
        }
        if (readyKernels.empty()) continue;
        
        /* sort in non-decreasing order */
        readyKernels.sort([](Kernel*& a, Kernel*& b){ return a->srcLayer->layerID < b->srcLayer->layerID; });

        /* print ready list */
#if (LOG_LEVEL >= VERBOSE)
        std::cout << "App " << app.appID << ": Ready kernel list: ";
        for (auto kernel : readyKernels)
        {
            std::cout << kernel->kernelID << ", ";
        }
        std::cout << std::endl;
#endif

        /* *******************************************************************
         * Pick the same kernel
         * *******************************************************************
         */
        memory_record[app->appID] += readyKernels.front()->srcLayer->getFilterMemory();
        for (auto k : readyKernels)
        {
            if (k->srcLayer->layerID == readyKernels.front()->srcLayer->layerID) 
            {
                kerenls_table[k->appID].push_back(k);
                memory_record[k->appID] += k->srcLayer->getIFMapMemory() + k->srcLayer->getOFMapMemory();
            }
        }
    }

    /* *******************************************************************
     * Launch kernel to gpu according to memory space
     * *******************************************************************
     */
    vector<pair<int, unsigned long long>> memory_budget = vector<pair<int, unsigned long long>> (memory_record.begin(), memory_record.end());
    sort(memory_budget.begin(), memory_budget.end(), [](auto& a, auto& b){ return a.second < b.second; });

    /* launch batch kernel and allocate memory */
    long long remaining_memory = system_resource.VRAM_SPACE;
    map<int, unsigned long long> memory_allocation;
    for (auto app_pair : memory_budget)
    {
        if (remaining_memory > app_pair.second)
        {
            remaining_memory -= app_pair.second;
        }
        else if (kerenls_table.find(app_pair.first) != kerenls_table.end())
        {
            auto layer = kerenls_table[app_pair.first].front()->srcLayer;
            
            /* Trace the batch size to fit the memory space */
            if (remaining_memory > layer->getMemoryUsage()) 
            {
                int batch_size = floor((remaining_memory - layer->getFilterMemory()) / (layer->getIFMapMemory() + layer->getOFMapMemory()));
                
                kerenls_table[app_pair.first].resize(batch_size);

                unsigned long long memory = (layer->getIFMapMemory() + layer->getOFMapMemory()) * kerenls_table[app_pair.first].size() + layer->getFilterMemory();
                app_pair.second   = memory;
                remaining_memory -= memory;
            }
        }
        else break;
    }

    for (auto kernel_pair : kerenls_table)
    {
        Kernel* kernel;
        if (kernel_pair.second.size() == 1) kernel = kernel_pair.second.front();
        else
        {
            vector<pair<Kernel*, int>> sync_kernels;
            unordered_set<int>* avaiable_sm = new unordered_set<int>;

            for (auto k : kernel_pair.second)
            {
                sync_kernels.push_back(make_pair(k, 1));
                avaiable_sm->insert(k->SM_List->begin(), k->SM_List->end());
            }
            kernel = new KernelGroup(sync_kernels);
            kernel->SM_List = move(avaiable_sm);
        }

        if (!kernel->SM_List->empty())
        {
            if (kernel->compileRequest(&mCPU->mMMU))
            {
                ASSERT(mCPU->mGPU->launchKernel(kernel), "Failed launch kernel");
                kernel->startCycle = total_gpu_cycle;
                kernel->running    = true;
            } 
            else log_I("compileRequest", "kernel: " + to_string(kernel->kernelID) + "has empty requests");
        }
    }
    
    return true;
}