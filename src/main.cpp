/**
 * \name    main.cpp
 * 
 * \brief   This code instance a GPGPU for simulating the GPGPU behavior
 * 
 * \date    APR 18, 2023
 */

#include "include/App_config.h"
#include "include/Log.h"

#include "include/GPGPU.hpp"

/* ************************************************************************************************
 * Main
 * ************************************************************************************************
 */
int main (int argc, char** argv)
{

    std::cout << "Hello GPGPU" << std::endl;
    
    GPGPU mGPGPU;

    mGPGPU.run();

    return 0;
}