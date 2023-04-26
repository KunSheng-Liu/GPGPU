/**
 * \name    GPU.hpp
 * 
 * \brief   Declare the structure of GPU
 * 
 * \note    Contains the Request format for GPU command
 * 
 * \date    APR 19, 2023
 */

#ifndef _GPU_HPP_
#define _GPU_HPP_

/* ************************************************************************************************
 * Include Library
 * ************************************************************************************************
 */
#include "App_config.h"
#include "Log.h"

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


#endif