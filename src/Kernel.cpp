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
Kernel::Kernel() : numOfRead(0), numOfWrite(0), numOfMemory(0), finish(false)
{
    requests = {};
    dependencyKernels = {};
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
Kernel::addRequest(Request request)
{
    numOfRead  += request.readAddresses.size();
    numOfWrite += request.writeAddresses.size();
    
    requests.push(request);
}