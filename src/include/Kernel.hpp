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

#include "GPU.hpp"
#include "Kernel.hpp"
#include "Layers.hpp"

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
    Kernel(int app_id, int kernel_id, Layer* src_layer, vector<Kernel*> dependencies);
    ~Kernel();


/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    void compileRequest (MMU* mmu);
    void addRequest (Request* request);
    void printInfo ();

    bool isReady();
    bool isFinish()  {return finish;}
    bool isRunning() {return running;}
    KernelInfo getKernelInfo() const {return info;}

/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
public:
    /* The index of source application. */
    const int appID;

    /* The index of kernel. Each model have a unique index */
    const int kernelID;

    bool finish;
    bool running;
    
private:

    /* source layer */
    Layer* srcLayer;

    KernelInfo info;

    queue<Request*> requests;

    vector<Kernel*> dependencyKernels; 
};

#endif