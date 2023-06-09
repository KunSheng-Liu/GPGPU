/**
 * \name    InferenceEngine.cpp
 * 
 * \brief   Implement the Kernel and Request.
 * 
 * \date    APR 19, 2023
 */

#include "include/Kernel.hpp"

/* ************************************************************************************************
 * Global Variable
 * ************************************************************************************************
 */
int Kernel::kernelCount = 0;

/** ===============================================================================================
 * \name    Kernel
 * 
 * \brief   The requests container for communicate between CPU to GPU.
 * 
 * \endcond
 * ================================================================================================
 */
// Kernel::Kernel() : numOfRead(0), numOfWrite(0), numOfMemory(0), finish(false)
// {
//     requests = {};
//     dependencyKernels = {};
// }


/** ===============================================================================================
 * \name    Kernel
 * 
 * \brief   The requests container for communicate between CPU to GPU.
 * 
 * \endcond
 * ================================================================================================
 */
Kernel::Kernel(int app_id, int model_id, Layer* src_layer, vector<Kernel*> dependencies) 
        : appID(app_id), modelID(model_id), kernelID(kernelCount++), srcLayer(src_layer), dependencyKernels(dependencies)
        , running(false), finish(false)
{
    requests = {};
}


/** ===============================================================================================
 * \name    ~Kernel
 * 
 * \brief   Destruct Kernel
 * 
 * \endcond
 * ================================================================================================
 */
Kernel::~Kernel()
{
    ASSERT(requests.empty(), "Error Destruct");
}


/** ===============================================================================================
 * \name    compileRequest
 * 
 * \brief   Compile the request by the source layer
 * 
 * \param   mmu     the memory management unit
 * 
 * \return  return True if the request not empty
 * 
 * \endcond
 * ================================================================================================
 */
bool
Kernel::compileRequest (MMU* mmu)
{
    srcLayer->memoryAllocate(mmu);

    srcLayer->Compile(mmu, this);

    kernelInfo.numOfMemory = srcLayer->getMemoryUsage();

#if PRINT_MODEL_DETIAL
    printInfo(true);
#endif
    return !requests.empty();
}


/** ===============================================================================================
 * \name    addRequest
 * 
 * \brief   Add a request into this kernel
 * 
 * \param   request     The GPU command
 * 
 * \endcond
 * ================================================================================================
 */
void
Kernel::addRequest(Request* request)
{
    request->requst_id = requests.size();
    for (auto access : request->readPages ) kernelInfo.numOfRead  += access.second;
    for (auto access : request->writePages) kernelInfo.numOfWrite += access.second;
    kernelInfo.numOfCycle += request->numOfInstructions;
    kernelInfo.numOfRequest++;
    
    requests.push(request);
}


/** ===============================================================================================
 * \name    accessRequest
 * 
 * \brief   Add a request into this kernel
 * 
 * \return  Request
 * 
 * \endcond
 * ================================================================================================
 */
Request*
Kernel::accessRequest()
{
    Request* request = move(requests.front());
    requests.pop();

    return move(request);
}


/** ===============================================================================================
 * \name    isReady
 * 
 * \brief   check whether the dependency kernels are all finished
 * 
 * \return  true or false
 * 
 * \endcond
 * ================================================================================================
 */
bool
Kernel::isReady()
{
    bool isReady = true;
    for (auto kernel : dependencyKernels) isReady &= kernel->isFinish();

    return isReady;
}


/** ===============================================================================================
 * \name    handleKernelCompletion
 * 
 * \brief   process and record the kernel status
 * 
 * \endcond
 * ================================================================================================
 */
void
Kernel::handleKernelCompletion()
{
    finish = true;
    running = false;
    log_W("Kernel", to_string(kernelID) + " (" + srcLayer->layerType + ") is finished [" + to_string(startCycle) + ", " + to_string(endCycle) + "]");

    /* *******************************************************************
     * Record the kernel information into file
     * *******************************************************************
     */
#if (PRINT_BLOCK_RECORD)
    ofstream file(LOG_OUT_PATH + program_name + ".txt", std::ios::app);
    file << "Finish kernel" << std::right << setw(4) << kernelID << ": [" << startCycle << ", " << endCycle << "]" << std::endl;
    for (auto& b_record : block_record)
    {
        file << "Finish block" << std::right << setw(5) << b_record.block_id << ": [" 
                << b_record.sm_id                 << ", "
                << b_record.start_cycle           << ", "
                << b_record.end_cycle             << ", "
                << b_record.launch_access_counter << ", "
                << b_record.return_access_counter << ", "
                << b_record.access_page_counter   << "]"
                << std::endl;
    #if (PRINT_WARP_RECORD)
        for (auto& w_record : b_record.warp_record)
        {
            file << std::right << setw(14) << "warp" << std::right << setw(3) << w_record.warp_id << ": ["
                    << w_record.start_cycle         << ", "
                    << w_record.end_cycle           << ", "
                    << w_record.computing_cycle     << ", "
                    << w_record.wait_cycle          << "]"
                    << std::endl;
        }
    #endif
    }
    file.close();
#endif
}


/** ===============================================================================================
 * \name    memoryRelease
 * 
 * \brief   Release no used memory space
 * 
 * \param   mmu     the memory management unit
 *  
 * \return  total page record of this kernel
 * 
 * \endcond
 * ================================================================================================
 */
PageRecord
Kernel::memoryRelease(MMU* mmu)
{
    ASSERT(requests.empty()); 

    dependencyKernels.clear(); 

    return srcLayer->memoryRelease(mmu);    
}


/** ===============================================================================================
 * \name    printInfo
 * 
 * \brief   Print the kernel information
 * 
 * \note    appID, kernelID, num of request, num of read, num of write, num of memory, num of cycle 
 * 
 * \endcond
 * ================================================================================================
 */
void
Kernel::printInfo(bool title)
{
    if (title) {
        std::cout << std::left << std::setw(10) << "appID"; 
        std::cout << std::left << std::setw(10) << "kernelID"; 
        std::cout << std::left << std::setw(10) << "Request"; 
        std::cout << std::left << std::setw(10) << "Read"; 
        std::cout << std::left << std::setw(10) << "Write"; 
        std::cout << std::left << std::setw(10) << "Memory"; 
        std::cout << std::left << std::setw(10) << "Cycle"; 
        std::cout << std::left << std::setw(14) << "Dependency"; 
        std::cout << std::left << std::setw(10) << "Finish"; 
        std::cout << std::endl;
    }
    

    std::cout << std::left << std::setw(10) << appID; 
    std::cout << std::left << std::setw(10) << kernelID; 
    std::cout << std::left << std::setw(10) << kernelInfo.numOfRequest; 
    std::cout << std::left << std::setw(10) << kernelInfo.numOfRead; 
    std::cout << std::left << std::setw(10) << kernelInfo.numOfWrite; 
    std::cout << std::left << std::setw(10) << kernelInfo.numOfMemory; 
    std::cout << std::left << std::setw(10) << kernelInfo.numOfCycle; 

    for (auto kernel : dependencyKernels)
    {
        std::cout << kernelID << ", "; 
    }

    std::cout << std::right << std::setw(15) << finish; 

    std::cout << std::endl;
}



/** ===============================================================================================
 * \name    KernelGroup
 * 
 * \brief   The container for merge multiple kernels
 * 
 * \endcond
 * ================================================================================================
 */
KernelGroup::KernelGroup(vector<pair<Kernel*, int>> kernels) : Kernel(kernels.front().first->appID, -1, nullptr, {}),  kernel_list(kernels)
{
    
}


/** ===============================================================================================
 * \name   ~KernelGroup
 * 
 * \brief   Destruct KernelGroup
 * 
 * \endcond
 * ================================================================================================
 */
KernelGroup::~KernelGroup()
{
    ASSERT(requests.empty(), "Error Destruct");
}


/** ===============================================================================================
 * \name    compileRequest
 * 
 * \brief   Compile the request by the source layer
 * 
 * \param   mmu     the memory management unit
 * 
 * \return  return True if the request not empty
 * 
 * \endcond
 * ================================================================================================
 */
bool
KernelGroup::compileRequest (MMU* mmu)
{
    /* *******************************************************************
     * Compile each kernel with same filter
     * *******************************************************************
     */
    auto filter = kernel_list.front().first->srcLayer->getFilter();
    for (int i = 1; i < kernel_list.size(); i++) kernel_list[i].first->srcLayer->setFilter(filter);
    
    for (auto kernel : kernel_list)
    {
        kernel.first->compileRequest(mmu);
        kernelInfo += kernel.first->kernelInfo;
        kernel.first->startCycle = total_gpu_cycle;
        kernel.first->running    = true;
    }

    /* *******************************************************************
     * Concat the requests of each kernel
     * *******************************************************************
     */
    while (!kernel_list.back().first->requests.empty())
    {
        for (auto kernel : kernel_list)
        {
            for (int i = 0; i < kernel.second; i++)
            {
                kernel.first->requests.front()->requst_id = requests.size();
                requests.push(move(kernel.first->requests.front()));
                kernel.first->requests.pop();
            }
        }
    }

    for (auto kernel : kernel_list) ASSERT(kernel.first->requests.empty(), "Fail to concat request");

    recorder = new RuntimeRecord;

    return !requests.empty();
}


/** ===============================================================================================
 * \name    handleKernelCompletion
 * 
 * \brief   process and record the kernel status for all kernels
 * 
 * \endcond
 * ================================================================================================
 */
void
KernelGroup::handleKernelCompletion()
{
    finish = true;
    running = false;

    string buff = "[";
    for (auto kernel : kernel_list) 
    {
        buff += to_string(kernel.first->kernelID) + ", ";
        kernel.first->finish  = true;
        kernel.first->running = false;
    //    *kernel.first->recorder += *recorder;
    }
    buff += "] (";
    buff += kernel_list.front().first->srcLayer->layerType;
    buff += ") is finished ";

    log_W("kernelGroup", buff + "[" + to_string(startCycle) + ", " + to_string(endCycle) + "]");

    /* *******************************************************************
     * Record the kernel information into file
     * *******************************************************************
     */
#if (PRINT_BLOCK_RECORD)
    ofstream file(LOG_OUT_PATH + program_name + ".txt", std::ios::app);
    file << buff << ": [" << startCycle << ", " << endCycle << "]" << std::endl;
    for (auto& b_record : block_record)
    {
        file << "Finish block" << std::right << setw(5) << b_record.block_id << ": [" 
                << b_record.sm_id                 << ", "
                << b_record.start_cycle           << ", "
                << b_record.end_cycle             << ", "
                << b_record.launch_access_counter << ", "
                << b_record.return_access_counter << ", "
                << b_record.access_page_counter   << "]"
                << std::endl;
    #if (PRINT_WARP_RECORD)
        for (auto& w_record : b_record.warp_record)
        {
            file << std::right << setw(14) << "warp" << std::right << setw(3) << w_record.warp_id << ": ["
                    << w_record.start_cycle         << ", "
                    << w_record.end_cycle           << ", "
                    << w_record.computing_cycle     << ", "
                    << w_record.wait_cycle          << "]"
                    << std::endl;
        }
    #endif
    }
    file.close();
#endif

    delete recorder;
    delete SM_List;
    delete this;
}