/**
 * \name    Approach_LazyB.cpp
 * 
 * \brief   Implement the function of related work \b Lazy_Batching used in CPU.hpp.
 * 
 * \details As a cloud server, the tasks is launched form the edge devices. Therefore try to maximize the batch size of the task 
 *          in kernel level.
 * 
 * \note    In this scenario, no memory limitation to the system.
 * \note    This approach use ResNet model
 * \note    Max batch size is constrained as 64
 * \note    The memory access overhead is set as 100 cycle
 * 
 * \date    May 25, 2023
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
Scheduler_LazyB::Inference_Admission ()
{  
    log_T("CPU", "Inference_Admission: LazyB");
    ASSERT(mCPU->mAPPs.size() == 1, "LazyB can only run one application");

    /* *******************************************************************
     * Check the GPU is idle (kernel is finished)
     * *******************************************************************
     */
    unordered_set<int> available_sm = mCPU->mGPU->getIdleSMs();
    if (available_sm.size() < GPU_SM_NUM) return false;

    /* *******************************************************************
     * Check the models haven't miss deadline, if so, terminate model
     * *******************************************************************
     */
    missDeadlineHandler ();
    
    /* *******************************************************************
     * Sort the models
     * *******************************************************************
     */
    auto app = mCPU->mAPPs.front();
    app->runningModels.sort([](Model*& a, Model*& b){
        return a->findReadyKernels().front()->kernelID > b->findReadyKernels().front()->kernelID;
    });

#if (PRINT_LAZY_BATCHING)
    for (auto model : app->runningModels)
    {
        cout << "Model " << model->modelID << " with " << model->getBatchSize() << " batch size: Ready kernel list: ";
        for (auto kernel : model->findReadyKernels()) cout << kernel->srcLayer->layerID << ", ";
        cout << endl;
    }
#endif

    /* *******************************************************************
     * Check the remaining slack time, and allocate SM
     * *******************************************************************
     */
    
    int batch_budget = LAZYB_MAX_BATCH_SIZE;
    unsigned long long slack_time = app->runningModels.back()->task.deadLine - total_gpu_cycle;

    for (auto rmodel = app->runningModels.rbegin(); rmodel != app->runningModels.rend(); rmodel++)
    {
        auto kernel_stat =  (*rmodel)->getKernelStatus();
        for (int i = 0; i < (*rmodel)->getNumOfLayer(); i++) if (!kernel_stat[i]) slack_time -= (*rmodel)->getBatchSize() * layerCycles[i];

        batch_budget -= (*rmodel)->getBatchSize();
        if (slack_time >= 0 && batch_budget >= 0) (*rmodel)->SM_budget = available_sm;
        else (*rmodel)->SM_budget = {};
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
Scheduler_LazyB::Kernel_Scheduler ()
{  
    log_T("CPU", "Kernel_Scheduler: LazyB");
    /* *******************************************************************
     * Assign SM to each application according to its model needed pages
     * *******************************************************************
     */
    unordered_set<int> available_sm = mCPU->mGPU->getIdleSMs();
    if (available_sm.size() < GPU_SM_NUM || mCPU->mAPPs.front()->runningModels.empty()) return false;

    auto app = mCPU->mAPPs.front();

    /* *******************************************************************
     * Perform merge
     * *******************************************************************
     */
    vector<pair<Kernel*, int>> sync_kernels;
    unordered_set<int>* avaiable_sm = new unordered_set<int>;

    int latest_layer_id;
    auto model = app->runningModels.begin();

    while (model != app->runningModels.end())
    {   
        if (!(*model)->SM_budget.empty())
        {
            latest_layer_id = (*model)->findReadyKernels().front()->srcLayer->layerID;
            break;
        }
        model++;
    }

    while (model != app->runningModels.end())
    {
        auto kernel = (*model)->findReadyKernels().front();
        if (kernel->srcLayer->layerID == latest_layer_id)
        {
            sync_kernels.push_back(make_pair(kernel, (*model)->getBatchSize()));
            avaiable_sm->insert(kernel->SM_List->begin(), kernel->SM_List->end());
        }
        model++;
    }

    /* *******************************************************************
     * Launch kernel to GPU
     * *******************************************************************
     */
    Kernel* kernel = (sync_kernels.size() == 1) ? sync_kernels.front().first : new KernelGroup(sync_kernels);
    
    kernel->SM_List = move(avaiable_sm);
    ASSERT(!kernel->SM_List->empty());

    if (kernel->compileRequest(&mCPU->mMMU))
    {
        ASSERT(mCPU->mGPU->launchKernel(kernel), "Failed launch kernel");
        kernel->startCycle = total_gpu_cycle;
        kernel->running    = true;
        
    } else {
        
        log_I("compileRequest", "kernel: " + to_string(kernel->kernelID) + "has empty requests");
    }
    

    return false;
}