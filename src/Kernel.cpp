/**
 * \name    InferenceEngine.cpp
 * 
 * \brief   Implement the Kernel and Request.
 * 
 * \date    APR 19, 2023
 */

#include "include/Kernel.hpp"

/** ===============================================================================================
 * \name    Kernel
 * 
 * \brief   The engine contain the inference scheduler and the gpu driver.
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
 * \brief   The engine contain the inference scheduler and the gpu driver.
 * 
 * \endcond
 * ================================================================================================
 */
Kernel::Kernel(int app_id, int kernel_id, Layer* src_layer, vector<Kernel*> dependencies) 
        : appID(app_id), kernelID(kernel_id), srcLayer(src_layer), dependencyKernels(dependencies)
        , running(false), finish(false)
{
    requests = {};
}


/** ===============================================================================================
 * \name    ~Kernel
 * 
 * \brief   Destruct InferenceEngine
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
    kernelInfo.numOfRead  += request->readPages.size();
    kernelInfo.numOfWrite += request->writePages.size();
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
    for (auto& kernel : dependencyKernels)
    {
        isReady &= kernel->isFinish();
    }
    return isReady;
}


/** ===============================================================================================
 * \name    release
 * 
 * \brief   Release no used memory space
 *  
 * \endcond
 * ================================================================================================
 */
void
Kernel::release()
{
    ASSERT(requests.empty());  
    srcLayer->release();
    dependencyKernels.clear(); 
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

    for (auto& kernel : dependencyKernels)
    {
        std::cout << kernel->kernelID << ", "; 
    }

    std::cout << std::right << std::setw(15) << finish; 

    std::cout << std::endl;
}