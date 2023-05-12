/**
 * \name    MemoryController.cpp
 * 
 * \brief   Implement the memory controller.
 * 
 * \date    APR 7, 2023
 */

#include "include/MemoryController.hpp"

/** ===============================================================================================
 * \name    MemoryController
 * 
 * \brief   Implement the memory controller for handling the uniquie physicall address.
 * 
 * \param   storage_limit       the phsical storage bound
 * \param   page_size           the size of one page frame
 * 
 * \endcond
 * ================================================================================================
 */
MemoryController::MemoryController(unsigned long long storage_limit, int page_size) : storageLimit(storage_limit), pageFrameOffset(log2(page_size))
{
    for (int i = 0; i < DRAM_SPACE / PAGE_SIZE; i++)
    {
        createPage();
    }

#if (PRINT_MEMORY_ALLOCATION)
    printInfo();
#endif

}


/** ===============================================================================================
 * \name    ~MemoryController
 * 
 * \brief   Destruct MemoryController
 * 
 * \endcond
 * ================================================================================================
 */
MemoryController::~MemoryController()
{

}


/** ===============================================================================================
 * \name    cycle
 * 
 * \brief   Handling the MemoryController in a period of cycle
 * 
 * \endcond
 * ================================================================================================
 */
void
MemoryController::cycle()
{
    log_I("MemoryController Cycle", to_string(total_gpu_cycle));
    mc_to_gmmu_queue.splice(mc_to_gmmu_queue.end(), gmmu_to_mc_queue);
}


/** ===============================================================================================
 * \name    createPage
 * 
 * \brief   Create new page into availablePageList if not excess the storageLimit.
 * 
 * \endcond
 * ================================================================================================
 */
void
MemoryController::createPage()
{
    ASSERT(pageIndex << pageFrameOffset <= storageLimit, "Cannot create anymore physical page");

    mPages.insert(make_pair(pageIndex, Page(pageIndex)));
    availablePageList.push_back(&mPages[pageIndex++]);

}


/** ===============================================================================================
 * \name    memoryAllocate
 * 
 * \brief   Pre-build some page frame for the DRAM needed.
 * 
 * \param   numOfByte     number of byte needs to be allocated.
 * 
 * \note    This API not consider the page reuse, page reclaiming. Each page will be only used in
 *          once, and therefore the allocation will be continuoused.
 * 
 * \return  the pointer of head physical page
 * 
 * \endcond
 * ================================================================================================
 */
Page*
MemoryController::memoryAllocate (int numOfByte)
{
    ASSERT(numOfByte != 0, "Try to allocate memory to empty data");

    Page* headPage;
    Page* prevPage;
    for (int i = 0; i < ceil((double)numOfByte / PAGE_SIZE); i++)
    {
        if (availablePageList.empty()){
            log_I("MemoryController::memoryAllocate()", "Out of DRAM Size");
            createPage();
        }

        usedPageList.push_back(availablePageList.front());
        availablePageList.pop_front();
        Page* tempPage = usedPageList.back();

        if(i == 0) {
            headPage = tempPage;
        } else {
            prevPage->nextPage = tempPage;
        }
        prevPage = tempPage;
    }

#if (PRINT_MEMORY_ALLOCATION)
    prevPage = headPage;
    std::cout << "Physical Pages: ";
    while(prevPage != nullptr) {
        std::cout << prevPage->pageIndex << ", ";
        prevPage = prevPage->nextPage;
    }
    std::cout << std::endl;
#endif

    return headPage;
}


/** ===============================================================================================
 * \name    printInfo
 * 
 * \brief   Print out the configuration.
 * 
 * \endcond
 * ================================================================================================
 */
void
MemoryController::printInfo()
{
    std::cout << "Memory Controller:" << std::endl;
    std::cout << std::right << std::setw(24) << "Storage Bound: "      << storageLimit << std::endl;
    std::cout << std::right << std::setw(24) << "PageFrame Offset: "   << pageFrameOffset << std::endl;
    std::cout << std::right << std::setw(24) << "AvailablePage Size: " << availablePageList.size() << std::endl;
    std::cout << std::right << std::setw(24) << "Used Size: "          << usedPageList.size() << std::endl;
}