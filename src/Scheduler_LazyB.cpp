/**
 * \name    Approach_LazyB.cpp
 * 
 * \brief   Implement the callback function  for related work \b Lazy_Batching used in CPU.hpp.
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
 * \endcond
 * ================================================================================================
 */
bool 
Scheduler_LazyB::Inference_Admission ()
{  
    log_T("CPU", "Inference_Admission: LazyB");

    /* *******************************************************************
     * Check the GPU is idle (kernel is finished)
     * *******************************************************************
     */
    list<int> available_sm = mCPU->mGPU->getIdleSMs();
    if (mCPU->mAPPs.front()->tasks.empty() || available_sm.size() < GPU_SM_NUM) return false;

    ASSERT(mCPU->mAPPs.size() == 1, "LazyB can only run one application");
    auto app = mCPU->mAPPs.front();

    /* *******************************************************************
     * Get the remaining slack time
     * *******************************************************************
     */
    int batch_budget = LAZYB_MAX_BATCH_SIZE;
    unsigned long long slack_time = (app->runningModels.empty()) ? LAZYB_CYCLE_DEADLINE : app->runningModels.back()->deadline - total_gpu_cycle;

    for (auto model : app->runningModels) 
    {
        auto kernel_stat = model->getKernelStatus();
        for (int i = 0; i < model->getNumOfLayer(); i++)
        {
            if (!kernel_stat[i]) slack_time -= model->getBatchSize() * layerCycles[i];
        }
        
        batch_budget -= model->getBatchSize();
    }

    /* *******************************************************************
     * Collect tasks from applications which can meet SLA Prediction
     * *******************************************************************
     */
    list<Application::Task> tasks;

    int task_size = min3(app->tasks.size() ,batch_budget, (int) floor(slack_time / modelCycle));
    for (int i = 0; i < task_size && !app->tasks.empty(); i++)
    {
        if (app->tasks.front().deadLine > total_gpu_cycle)
        {
            tasks.push_back(app->tasks.front());
        }
        app->tasks.pop();
    }

    /* *******************************************************************
     * Launch new model
     * *******************************************************************
     */
    if (!tasks.empty())
    {
        app->runningModels.emplace_front(new Model(app->appID, app->modelType, app->inputSize, tasks.size()));
                    
        Model* model = app->runningModels.front();

        model->SM_budget = available_sm;

        model->buildLayerGraph();
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
    if (mCPU->mAPPs.front()->runningModels.empty()) return false;
    /* *******************************************************************
     * Assign SM to each application according to its model needed pages
     * *******************************************************************
     */
    list<int> available_sm = mCPU->mGPU->getIdleSMs();
    if (available_sm.size() < GPU_SM_NUM) return false;

    auto app = mCPU->mAPPs.front();

    /* *******************************************************************
     * Sort the models
     * *******************************************************************
     */
    app->runningModels.sort([](Model*& a, Model*& b){
        return a->findReadyKernels().front()->kernelID > b->findReadyKernels().front()->kernelID;
    });

    for (auto model : app->runningModels)
    {
        cout << "Model " << model->modelID << " with " << model->getBatchSize() << " batch size: Ready kernel list: ";
        for (auto kernel : model->findReadyKernels()) 
        {
            cout << kernel->srcLayer->layerID << ", ";
        }
        cout << endl;
    }

    /* *******************************************************************
     * Perform merge
     * *******************************************************************
     */
    vector<pair<Kernel*, int>> sync_kernels;

    int latest_layer_id = app->runningModels.front()->findReadyKernels().front()->srcLayer->layerID;
    for (auto model : app->runningModels)
    {
        if (model->findReadyKernels().front()->srcLayer->layerID == latest_layer_id)
        {
            sync_kernels.push_back(make_pair(model->findReadyKernels().front(), model->getBatchSize()));
        }
    }

    /* *******************************************************************
     * Launch kernel to GPU
     * *******************************************************************
     */
    Kernel* kernel = (sync_kernels.size() == 1) ? sync_kernels.front().first : new KernelGroup(sync_kernels);

    kernel->SM_List = &app->runningModels.front()->SM_budget;
    ASSERT(!kernel->SM_List->empty());

    if (kernel->compileRequest(&mCPU->mMMU))
    {
        kernel->running = mCPU->mGPU->launchKernel(kernel);
        if (kernel->running) kernel->start_cycle = total_gpu_cycle;
        
    } else {
        
        log_I("compileRequest", "kernel: " + to_string(kernel->kernelID) + "has empty requests");
    }
    

    return false;
}