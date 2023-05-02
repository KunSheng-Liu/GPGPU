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
    srcLayer->issueLayer(mmu, this);

    info.numOfMemory = srcLayer->getMemoryUsage();

#if PRINT_MODEL_DETIAL
    printInfo();
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
    info.numOfRead  += request->readPages.size();
    info.numOfWrite += request->writePages.size();
    info.numOfCycle += request->numOfInstructions;
    info.numOfRequest++;
    
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
    Request* request = requests.front();
    requests.pop();

    return request;
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
    for (auto kernel : dependencyKernels)
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
Kernel::printInfo()
{
    std::cout << std::left << std::setw(10) << appID; 
    std::cout << std::left << std::setw(10) << kernelID; 
    std::cout << std::left << std::setw(10) << info.numOfRequest; 
    std::cout << std::left << std::setw(10) << info.numOfRead; 
    std::cout << std::left << std::setw(10) << info.numOfWrite; 
    std::cout << std::left << std::setw(10) << info.numOfMemory; 
    std::cout << std::left << std::setw(10) << info.numOfCycle; 
    for (auto kernel : dependencyKernels)
    {
        std::cout << std::left << std::setw(3) << kernel->kernelID; 
    }

    std::cout << std::right << std::setw(10) << finish; 

    std::cout << std::endl;
}