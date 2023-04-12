/**
 * \name    MemoryControl.hpp
 * 
 * \brief   Declare the memory API 
 * 
 * \date    APR 7, 2023
 */

#ifndef _MEMORYCONTROL_HPP_
#define _MEMORYCONTROL_HPP_

/* ************************************************************************************************
 * Include Library
 * ************************************************************************************************
 */
#include "App_config.h"
#include "Log.h"


/** ===============================================================================================
 * \name    MemoryControl
 * 
 * \brief   The class of memory controller for handling the uniquie physicall address.
 * 
 * \endcond
 * ================================================================================================
 */
class MemoryControl
{
/* ************************************************************************************************
 * Class Constructor
 * ************************************************************************************************
 */ 
public:
   MemoryControl(long long storage_limit, int page_size);

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:

    void init();

    pair<int, int> memoryAllocate (int numByte);

    int  createPage ();

    void printInfo();


/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
private:
    const long long storageLimit;
    const int pageFrameOffset;

    int pageIndex = 0;
    int physicalAddressCount = 0;       // unit in Byte

    queue<int> availablePageList;

};

#endif