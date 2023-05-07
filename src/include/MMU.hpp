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


/** ===============================================================================================
 * \name    TLB
 * 
 * \brief   The class of translation lookaside table for translating the VA to PA by hash table.
 * 
 * \note    This TLB use the Least Recently Used (LRU) algorithm. The table hold the pair of start 
 *          and end physical address.
 * 
 * \endcond
 * ================================================================================================
 */
class TLB
{
/* ************************************************************************************************
 * Class Constructor
 * ************************************************************************************************
 */ 
public:

    TLB(int _capacity) : capacity(_capacity) {}

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:

    pair<Page*, int> lookup (intptr_t va);
    void insert(intptr_t va, pair<Page*, int> pa_pair);
    
/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
private:
    struct PageTableEntry {
        intptr_t VA;
        pair<Page*, int> PAPair;
        PageTableEntry(intptr_t va, pair<Page*, int> pa_pair) : VA(va), PAPair(pa_pair) {}
    };

    const int capacity;
    std::list<PageTableEntry> list;
    unordered_map<intptr_t, std::list<PageTableEntry>::iterator> table;
};



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
    void memoryAllocate (intptr_t va, int numOfByte);
    vector<unsigned long long> addressTranslate (intptr_t va);

/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
private:
    MemoryController* mMC;
    TLB mTLB;
};

#endif