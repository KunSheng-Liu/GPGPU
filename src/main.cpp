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
    timeval start, end;
    gettimeofday(&start, NULL);

        std::cout << "Hello GPGPU" << std::endl;

        GPGPU mGPGPU;

        mGPGPU.run();

        std::cout << "GPGPU Done!" << std::endl;

    gettimeofday(&end, NULL);
    std::cout << "Total spend time: " << to_string((1000000 * (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)) * 0.001) << " ms" << std::endl;
    

    return 0;
}