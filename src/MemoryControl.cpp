/**
 * \name    MemoryControl.cpp
 * 
 * \brief   Implement the memory controller.
 * 
 * \date    APR 7, 2023
 */

#include "include/MemoryControl.hpp"

/** ===============================================================================================
 * \name    MemoryControl
 * 
 * \brief   Implement the memory controller for handling the uniquie physicall address.
 * 
 * \param   storage_limit       the phsical storage bound
 * \param   page_size           the size of one page frame
 * 
 * \endcond
 * ================================================================================================
 */
MemoryControl::MemoryControl(long long storage_limit, int page_size) : storageLimit(storage_limit), pageFrameOffset(log2(page_size))
{
    init();
    printInfo();
}


/** ===============================================================================================
 * \name    init
 * 
 * \brief   Pre-build some page frame for the DRAM needed.
 * 
 * \endcond
 * ================================================================================================
 */
void
MemoryControl::init()
{
    for (; pageIndex < DRAM_SPACE / PAGE_SIZE; pageIndex++)
    {
        availablePageList.push(pageIndex << pageFrameOffset);
    }
}


/** ===============================================================================================
 * \name    memoryAllocate
 * 
 * \brief   Pre-build some page frame for the DRAM needed.
 * 
 * \param   numOfByte     number of byte needs to be allocated.
 * 
 * \return  pair<int, int>, a pair of PA [start, end]
 * 
 * \endcond
 * ================================================================================================
 */
pair<int, int>
MemoryControl::memoryAllocate (int numOfByte)
{
    int startAddr = physicalAddressCount;
    physicalAddressCount += numOfByte - 1;

    return make_pair(startAddr, physicalAddressCount);
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
MemoryControl::printInfo()
{
    std::cout << "Memory Controller:" << std::endl;
    std::cout << std::right << std::setw(24) << "Storage Bound: " << storageLimit << std::endl;
    std::cout << std::right << std::setw(24) << "PageFrame Offset: " << pageFrameOffset << std::endl;
    std::cout << std::right << std::setw(24) << "AvailablePage Size: " << availablePageList.size() << std::endl;
}