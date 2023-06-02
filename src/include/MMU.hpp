/**
 * \name    MMU.hpp
 * 
 * \brief   Declare the structure of MMU and TLB
 * 
 * \note    In this simulator, the VA is the pointer of the data.
 * 
 * \date    APR 10, 2023
 */

#ifndef _MMU_HPP_
#define _MMU_HPP_

/* ************************************************************************************************
 * Include Library
 * ************************************************************************************************
 */
#include "App_config.h"
#include "Log.h"
#include "MemoryController.hpp"
#include "TLB.hpp"


/** ===============================================================================================
 * \name    MMU
 * 
 * \brief   The class of memory manage unit for handling the CPU virtual address to physical address.
 * 
 * \endcond
 * ================================================================================================
 */
class MMU
{
/* ************************************************************************************************
 * Class Constructor
 * ************************************************************************************************
 */ 
public:

    MMU(MemoryController* mc);

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    int  lookup (int va);
    bool memoryAllocate (int va, int numOfByte);
    void memoryRelease  (int va);
    vector<unsigned long long> addressTranslate (int va);

    PageRecord getPageSummary (int va);

/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
private:
    MemoryController* mMC;
    TLB<int, pair<Page*, int>> mTLB;
};

#endif