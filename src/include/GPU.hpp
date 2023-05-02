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

/* ************************************************************************************************
 * Type Define
 * ************************************************************************************************
 */
struct Request{
    int numOfInstructions;
    vector<int> readPages;
    vector<int> writePages;

    Request (vector<int> read_pages = {}, vector<int> write_pages = {}) 
        : numOfInstructions(0), readPages(read_pages), writePages(write_pages) {}
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
    list<SM> mSMs;

    list<Kernel*> runningKernels;
};


#endif