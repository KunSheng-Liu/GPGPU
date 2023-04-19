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
    vector<int> readAddresses;
    vector<int> writeAddresses;

    Request (vector<int> read_addresses, vector<int> write_addresses) 
        : readAddresses(read_addresses), writeAddresses(write_addresses) {}
};


#endif