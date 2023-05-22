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
 * \brief   The container of requests for communicate between CPU to GPU.
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
    int numOfRead    = 0;
    int numOfWrite   = 0;
    int numOfCycle   = 0;
    int numOfMemory  = 0;
    int numOfRequest = 0;
};

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    Request* accessRequest ();

    void addRequest (Request* request);
    bool compileRequest (MMU* mmu);
    PageRecord memoryRelease  (MMU* mmu);

    void printInfo (bool title = false);

    bool isReady();
    bool isFinish()  {return finish;}
    bool isRunning() {return running;}
    KernelInfo getKernelInfo() const {return kernelInfo;}

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

    unsigned long long start_cycle = 0;
    unsigned long long end_cycle = 0;

    Layer* srcLayer;

    KernelInfo kernelInfo;

    // RuntimeInfo* recorder;
    list<int>* SM_List;

    RuntimeRecord* recoder;

    list<Block::BlockRecord> block_record;

    queue<Request*> requests;

    vector<Kernel*> dependencyKernels; 

private:

    /* Number of kernel be created */
    static int kernelCount;
};

#endif