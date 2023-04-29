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
    int pageIndex;
    Page* nextPage;

    Page(int page_indnex = -1, Page* next_page = nullptr) : pageIndex(page_indnex), nextPage(next_page) {}
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
   MemoryController(long long storage_limit, int page_size);

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
    const long long storageLimit;
    const int pageFrameOffset;

    int pageIndex = 0;

    queue<Page*> availablePageList;
    queue<Page*> usedPageList;

};

#endif