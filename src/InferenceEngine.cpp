/**
 * \name    InferenceEngine.cpp
 * 
 * \brief   Implement the gpu driver.
 * 
 * \date    APR 17, 2023
 */

#include "include/InferenceEngine.hpp"

/** ===============================================================================================
 * \name    InferenceEngine
 * 
 * \brief   The engine contain the inference scheduler and the gpu driver.
 * 
 * \endcond
 * ================================================================================================
 */
InferenceEngine::InferenceEngine(MMU* mmu, vector<Application*>* apps) : mMMU(mmu), mAPPs(apps)
{

}

/** ===============================================================================================
 * \name   ~InferenceEngine
 * 
 * \brief   Destruct InferenceEngine
 * 
 * \endcond
 * ================================================================================================
 */
InferenceEngine::~InferenceEngine()
{

}


/** ===============================================================================================
 * \name    cycle
 * 
 * \brief   Handling the tasks in the task queue
 * 
 * \endcond
 * ================================================================================================
 */

void
InferenceEngine::cycle()
{
    log_D("InferenceEngine", "cycle");
    Dynamic_Batching_Algorithm();
    Kernek_Inference_Scheduler();
}


/** ===============================================================================================
 * \name    Dynamic_Batching_Algorithm
 * 
 * \brief   First method, to determine the batch size of each model by the given information.
 * 
 * \endcond
 * ================================================================================================
 */

void
InferenceEngine::Dynamic_Batching_Algorithm()
{
    log_D("InferenceEngine", "Dynamic_Batching_Algorithm");

    struct timeval start, end;
    gettimeofday(&start, NULL);

    /* Choose the batch size of each model and create the instance */
    for (auto app : *mAPPs)
    {
        int batchSize = 1;
        app->runningModels.emplace_back(new Model(batchSize));
        
        Model* model = app->runningModels.back();
        model->buildLayerGraph(app->modelType);
        model->memoryAllocate(mMMU);
        vector<Kernel>& kernels = model->compileToKernel();

        for (auto& kernel : kernels)
        {
            kernel.srcLayer->issueLayer(mMMU, &kernel);
        }
        for (auto& kernel : kernels)
        {
            cout << kernel.srcLayer->layerType << " requests num: " << kernel.requests.size() << endl;
        }
    }

    gettimeofday(&end, NULL);
    float spendTime = (1000000 * (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)) * 0.001;
    log_I("main", "Total spend: " + to_string(spendTime) + " ms");

    ASSERT(false);
}


/** ===============================================================================================
 * \name    Kernek_Inference_Scheduler
 * 
 * \brief   Second method, launch the model's kernel by the resource constrain.
 * 
 * \endcond
 * ================================================================================================
 */

void
InferenceEngine::Kernek_Inference_Scheduler()
{
    log_D("InferenceEngine", "Kernek_Inference_Scheduler");
}



