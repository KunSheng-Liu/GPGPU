/**
 * \name    GPU.hpp
 * 
 * \brief   Declare the structure of GPU
 * 
 * \note    Contains the Request format for GPU command
 * 
 * \date    APR 30, 2023
 */

#ifndef _GPU_HPP_
#define _GPU_HPP_

/* ************************************************************************************************
 * Include Library
 * ************************************************************************************************
 */
#include "App_config.h"
#include "Log.h"

#include "GMMU.hpp"
#include "Kernel.hpp"
#include "SM.hpp"


/** ===============================================================================================
 * \name    Request
 * 
 * \brief   The class of the Request
 * 
 * \endcond
 * ================================================================================================
 */
class Request{
/* ************************************************************************************************
 * Class Constructor
 * ************************************************************************************************
 */ 
public:

    Request (vector<int> read_pages = {}, vector<int> write_pages = {}) 
        : numOfInstructions(0), readPages(read_pages), writePages(write_pages) {}

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    int numOfInstructions;
    vector<int> readPages;
    vector<int> writePages;
    
};


/** ===============================================================================================
 * \name    GPU
 * 
 * \brief   The class of the GPU
 * 
 * \endcond
 * ================================================================================================
 */
class GPU
{
/* ************************************************************************************************
 * Class Constructor
 * ************************************************************************************************
 */ 
public:

    GPU(MemoryController* mc);

   ~GPU();

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    void cycle ();

    bool launchKernel (Kernel* kernel);

    bool idle() {return runningKernels.empty() && commandQueue.empty();}
    GMMU* getGMMU() {return &mGMMU;}

private:
    void Runtime_Block_Scheduling();
    void Check_Finish_Kernel();

/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
public:
    bool canIssueKernel;

    queue<Kernel*> commandQueue;
    list<Kernel*> finishedKernels;

private:
    MemoryController* mMC;
    GMMU mGMMU;
    map<int, SM> mSMs;

    list<Kernel*> runningKernels;
};


#endif