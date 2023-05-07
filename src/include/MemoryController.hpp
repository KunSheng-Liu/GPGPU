/**
 * \name    MemoryController.hpp
 * 
 * \brief   Declare the memory API 
 * 
 * \date    APR 7, 2023
 */

#ifndef _MEMORYCONTROLLER_HPP_
#define _MEMORYCONTROLLER_HPP_

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
struct Page {
    unsigned long long pageIndex;
    Page* nextPage;

    Page(unsigned page_index = -1, Page* next_page = nullptr) : pageIndex(page_index), nextPage(next_page) {}
};


/** ===============================================================================================
 * \name    MemoryController
 * 
 * \brief   The class of memory controller for handling the uniquie physicall address.
 * 
 * \endcond
 * ================================================================================================
 */
class MemoryController
{
/* ************************************************************************************************
 * Class Constructor
 * ************************************************************************************************
 */ 
public:
    MemoryController(unsigned long long storage_limit, int page_size);
    ~MemoryController();
/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:

    void init();
    
    void cycle ();

    Page* memoryAllocate (int numByte);

    void createPage ();

    void printInfo();


/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
private:
    const unsigned long long storageLimit;
    const unsigned pageFrameOffset;

    unsigned long long pageIndex = 0;

    queue<Page*> availablePageList;
    queue<Page*> usedPageList;

};

#endif