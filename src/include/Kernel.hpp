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

    Kernel();
    ~Kernel();


/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    void addRequest (Request* request);
    void checkDependency ();

/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
public:

    int kernelID;

    bool finish;

    int numOfRead;
    int numOfWrite;
    int numOfCycle;
    int numOfMemory;

    /* source layer */
    Layer* srcLayer;

    queue<Request*> requests;

    vector<Kernel*> dependencyKernels; 
};

#endif