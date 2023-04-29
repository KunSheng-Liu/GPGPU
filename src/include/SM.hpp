/**
 * \name    SM.hpp
 * 
 * \brief   Declare the structure of SM
 * 
 * \date    APR 30, 2023
 */

#ifndef _SM_HPP_
#define _SM_HPP_

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
struct SMInfo {
    unsigned long long exec_cycle;
	unsigned long long computing_cycle;
	unsigned long long wait_cycle;
	unsigned long long idle_cycle;
};

struct BlockInfo
{
	Kernel* running_kernel;
	//bool suspend;
	unsigned long long launch_access_counter = 0;
	unsigned long long return_access_counter = 0;
	unsigned launch_warp_counter = 0;
	//unsigned long long computing_time = 0;
	// according diff resource, kernel in different sm can access warp number is not same
	unsigned max_can_launch_warp_number = 0;
	unsigned bind_block_number = 0;

	// launch information 
	list<int> wait_cmputing_time;
	// map<unsigned, launch_request_info* > launch_info;
	bool unbind_flag = false;
};


/** ===============================================================================================
 * \name    SM
 * 
 * \brief   Contains the model and it's data
 * 
 * \endcond
 * ================================================================================================
 */
class SM
{
/* ************************************************************************************************
 * Class Constructor
 * ************************************************************************************************
 */ 
public:

    SM();

   ~SM();

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    void cycle ();
    bool isRunning();
    bool isComputing();
    bool checkIsComplete();

    void setGMMU (GMMU* gmmu) {mGMMU = gmmu;}
    
/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
public:
    const int smID;

private:
    static int SMCount;

    GMMU* mGMMU;

    SMInfo info;

};

#endif