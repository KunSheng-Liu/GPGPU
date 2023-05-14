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

#include "Memory.hpp"

/* ************************************************************************************************
 * Type Define
 * ************************************************************************************************
 */
typedef enum {
    SPACE_NONE,
    SPACE_VRAM,
	SPACE_DRAM,
}Page_Location;

struct PageRecord {
    unsigned long long write_counter = 0;
	unsigned long long read_counter = 0;

    unsigned long access_count = 0;
    unsigned long swap_count = 0;
};

struct Page {
    unsigned long long pageIndex;
    Page_Location location;
    PageRecord record;
    Page* nextPage;

    Page(unsigned long long page_index = -1, Page_Location location = SPACE_NONE, Page* next_page = nullptr) : pageIndex(page_index), location(location), nextPage(next_page) {}
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
    
    void cycle ();

    Page* refer (unsigned long long page_id) {return &mPages[page_id];}

    Page* memoryAllocate (int numByte);

    void printInfo();

private:

    void createPage ();

/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
private:
    const unsigned long long storageLimit;
    const unsigned pageFrameOffset;

    unsigned long long pageIndex = 0;

    map<unsigned long long , Page> mPages;

    list<Page*> availablePageList;
    list<Page*> usedPageList;

    list<MemoryAccess*> gmmu_to_mc_queue;
	list<MemoryAccess*> mc_to_gmmu_queue;

friend GMMU;
};

#endif