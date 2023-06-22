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
    Request (vector<pair<unsigned long long, int>> read_pages = {}, vector<pair<unsigned long long, int>> write_pages = {}, int numOfInstructions = 0) 
        : numOfInstructions(numOfInstructions), readPages(read_pages), writePages(write_pages) {}

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    int requst_id;
    int numOfInstructions;
    
    /* *******************************************************************
     * \param first     start physical address
     * \param second    number of access to this physical address
     * *******************************************************************
    */
    vector<pair<unsigned long long, int>> readPages, writePages;
    
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
    bool terminateModel (int app_id, int model_id);
    void statistic();

    bool isIdle() {return runningKernels.empty() && commandQueue.empty();}
    GMMU* getGMMU() {return &mGMMU;}
    unordered_set<int> getIdleSMs();

private:
    void Runtime_Block_Scheduling();
    void Check_Finish_Kernel();

/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
public:
    bool canIssueKernel;

    list<Kernel*>  commandQueue = {};
    list<Kernel*>  runningKernels = {};
    list<Kernel*>  finishedKernels = {};

private:
    GMMU mGMMU;

    MemoryController* mMC;

    map<int, SM> mSMs;

friend GMMU;
};

#endif