/**
 * \name    MMU.hpp
 * 
 * \brief   Declare the structure of MMU and TLB
 * 
 * \note    In this simulator, the VA is the pointer of the data.
 * 
 * \date    APR 6, 2023
 */

#ifndef _MMU_HPP_
#define _MMU_HPP_

/* ************************************************************************************************
 * Include Library
 * ************************************************************************************************
 */
#include "App_config.h"
#include "Log.h"
#include "MemoryControl.hpp"


/* ************************************************************************************************
 * Pre-declaration Class
 * ************************************************************************************************
 */
class TLB;

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

    MMU(MemoryControl* mc);

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    void memoryAllocate (int va, int numOfByte);
    pair<int, int> addressTranslate (int va);

/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
private:
    MemoryControl* mMC;
    TLB* mTLB;
};



/** ===============================================================================================
 * \name    TLB
 * 
 * \brief   The class of translation lookaside table for translating the VA to PA by hash table.
 * 
 * \note    This TLB use the Least Recently Used (LRU) algorithm.
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

    pair<int, int> lookup (int va);
    void insert(int va, pair<int, int> pa_pair);
    
/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
private:
    struct PageTableEntry {
        int VA;
        pair<int, int> PAPair;
        PageTableEntry(int va, pair<int, int> pa_pair) : VA(va), PAPair(pa_pair) {}
    };

    const int capacity;
    std::list<PageTableEntry> list;
    unordered_map<int, std::list<PageTableEntry>::iterator> table;
};

#endif