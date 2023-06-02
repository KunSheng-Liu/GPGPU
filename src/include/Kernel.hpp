/**
 * \name    Kernel.hpp
 * 
 * \brief   Declare the Kernel as the container of CPU to GPU Request.
 * 
 * \date    APR 19, 2023
 */

#ifndef _KERNEL_HPP_
#define _KERNEL_HPP_

/* ************************************************************************************************
 * Include Library
 * ************************************************************************************************
 */
#include "App_config.h"
#include "Log.h"

#include "Block.hpp"
#include "GPU.hpp"
#include "Layers.hpp"


/** ===============================================================================================
 * \name    Kernel
 * 
 * \brief   The requests container for communicate between CPU to GPU.
 * 
 * \endcond
 * ================================================================================================
 */
class Kernel
{
/* ************************************************************************************************
 * Class Constructor
 * ************************************************************************************************
 */ 
public:

    // Kernel();
    Kernel(int app_id, int model_id, Layer* src_layer, vector<Kernel*> dependencies);
   ~Kernel();

/* ************************************************************************************************
 * Type Define
 * ************************************************************************************************
 */
struct KernelInfo {
    unsigned long long numOfRead    = 0;
    unsigned long long numOfWrite   = 0;
    unsigned long long numOfCycle   = 0;
    unsigned long long numOfMemory  = 0;
    unsigned long long numOfRequest = 0;

    KernelInfo& operator+= (const KernelInfo& other) {
        numOfRead    += other.numOfRead;
        numOfWrite   += other.numOfWrite;
        numOfCycle   += other.numOfCycle;
        numOfMemory  += other.numOfMemory;
        numOfRequest += other.numOfRequest;
        return *this;
    }
};

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    virtual bool compileRequest (MMU* mmu);
    virtual void handleKernelCompletion ();

    void addRequest (Request* request);
    Request* accessRequest ();

    PageRecord memoryRelease  (MMU* mmu);

    bool isReady();
    bool isFinish()  {return finish;}
    bool isRunning() {return running;}
    KernelInfo getKernelInfo() const {return kernelInfo;}

    void printInfo (bool title = false);

/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
public:
    /* The index of source application. */
    const int appID;

    /* The index of source model. */
    const int modelID;

    /* The index of kernel. Each model have a unique index */
    const int kernelID;

    bool finish;
    bool running;

    unsigned long long startCycle, endCycle;

    Layer* srcLayer;

    KernelInfo kernelInfo;

    // RuntimeInfo* recorder;
    unordered_set<int>* SM_List;

    RuntimeRecord* recorder;

    list<Block::BlockRecord> block_record;

    queue<Request*> requests;

    vector<Kernel*> dependencyKernels; 

private:

    /* Number of kernel be created */
    static int kernelCount;
};




/** ===============================================================================================
 * \name    KernelGroup
 * 
 * \brief   The container for merge multiple kernels
 * 
 * \endcond
 * ================================================================================================
 */
class KernelGroup : public Kernel
{
/* ************************************************************************************************
 * Class Constructor
 * ************************************************************************************************
 */ 
public:

    KernelGroup(vector<pair<Kernel*, int>> kernels);
   ~KernelGroup();

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    bool compileRequest (MMU* mmu) override;
    void handleKernelCompletion () override;

/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
private:
    vector<pair<Kernel*, int>> kernel_list;

};

#endif