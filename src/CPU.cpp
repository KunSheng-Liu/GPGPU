/**
 * \name    CPU.cpp
 * 
 * \brief   Implement the CPU and it's components.
 * 
 * \date    APR 6, 2023
 */

#include "include/CPU.hpp"

/** ===============================================================================================
 * \name    CPU
 * 
 * \brief   The class of the CPU, contains MMU, TLB.
 * 
 * \param   mc      the pointer of memory controller
 * \param   gpu     the pointer of GPU
 * 
 * \endcond
 * ================================================================================================
 */
CPU::CPU(MemoryController* mc, GPU* gpu) : mMC(mc), mGPU(gpu), mMMU(MMU(mc))
{

#if (TASK_SET == TEST)
    mAPPs.push_back(new Application ((char*)"Test"));
    mAPPs.push_back(new Application ((char*)"VGG16"));
    mAPPs.push_back(new Application ((char*)"ResNet18"));
#elif (TASK_SET == LIGHT)
#elif (TASK_SET == HEAVY)
#elif (TASK_SET == MIX)
#endif

    
}


/** ===============================================================================================
 * \name    CPU
 * 
 * \brief   Destruct CPU
 * 
 * \endcond
 * ================================================================================================
 */
CPU::~CPU()
{

}


/** ===============================================================================================
 * \name    cycle
 * 
 * \brief   Handling the CPU in a period of cycle
 * 
 * \endcond
 * ================================================================================================
 */
void
CPU::cycle()
{
    log_I("CPU Cycle", to_string(total_gpu_cycle));

    Check_Finish_Kernel();

    Kernek_Inference_Scheduler();
    
    Dynamic_Batching_Algorithm();

    /* check new task */
    for (auto app: mAPPs)
    {
        app->cycle();
    }
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
CPU::Dynamic_Batching_Algorithm()
{
    log_D("CPU", "Dynamic_Batching_Algorithm");

    /* Choose the batch size of each model and create the instance */
    for (auto app : mAPPs)
    {
        if (app->tasks.size() != 0)
        {
            int batchSize = 1;
            Task task = app->tasks.front();
            app->tasks.pop();

            app->runningModels.emplace_back(new Model(app->appID, batchSize));
            
            Model* model = app->runningModels.back();
            model->buildLayerGraph(app->modelType);
            model->memoryAllocate(&mMMU);
        }
        
    }
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
CPU::Kernek_Inference_Scheduler()
{
    log_D("CPU", "Kernek_Inference_Scheduler");

    // handle the kernel dependency, and launch next kernel

    list<Kernel*> readyKernels;
    for (auto app : mAPPs)
    {
        for (auto model : app->runningModels)
        {
            readyKernels.splice(readyKernels.end(), model->findReadyKernels());
        }
    }

    /* print ready list */
    std::cout << "Ready kernel list: ";
    for (auto kernel : readyKernels)
    {
        std::cout << kernel->kernelID << ", ";
    }
    std::cout << endl;

    /* launch kernel into gpu */
    for (auto kernel : readyKernels)
    {
        if (kernel->compileRequest(&mMMU))
        {
            kernel->running = mGPU->launchKernel(kernel);
            
        } else {
            
            log_W("compileRequest", "kernel: " + to_string(kernel->kernelID) + "has empty requests");
        }
    }    
}


/** ===============================================================================================
 * \name    Check_Finish_Kernel
 * 
 * \brief   Check whether the kernel has been finished, Record kernel inside finishedKernels.
 * 
 * \endcond
 * ================================================================================================
 */
void
CPU::Check_Finish_Kernel()
{  
    /* check finish kernel */
    if (!mGPU->finishedKernels.empty())
    {
        auto kernel = mGPU->finishedKernels.front();
        mGPU->finishedKernels.pop_front();

        log_I("", "Kernel: " + to_string(kernel->kernelID) + " is finished");
        kernel->finish = true;
        kernel->running = false;

        /* recording... */

        /* release */
        kernel->release();
        
    }

    for (auto app : mAPPs)
    {
        app->runningModels.remove_if([](Model* m){return m->checkFinish();});
    }
}


/** ===============================================================================================
 * \name    Check_All_Applications_Finish
 * 
 * \brief   Check whether the applications has finished.
 * 
 * \endcond
 * ================================================================================================
 */
bool
CPU::Check_All_Applications_Finish()
{  
    bool finish = true;
    for (auto app : mAPPs)
    {
        finish &= app->finish;
    }
    return finish;
}
