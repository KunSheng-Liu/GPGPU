/**
 * \name    MMU.cpp
 * 
 * \brief   Implement the MMU and TLB.
 * 
 * \date    APR 10, 2023
 */

#include "include/MMU.hpp"


/** ===============================================================================================
 * \name    MMU
 * 
 * \brief   The memory manage unit for handling the CPU virtual address to physical address.
 * 
 * \param   mc      the pointer of Memory Controller
 * 
 * \endcond
 * ================================================================================================
 */
MMU::MMU(MemoryController* mc): mMC(mc), mTLB(TLB<int, pair<Page*, unsigned long long>>(system_resource.DRAM_SPACE / PAGE_SIZE))
{

}


/** ===============================================================================================
 * \name    lookup
 * 
 * \brief   Lookup the whether the virtial address is exist
 * 
 * \param   va          the virtual address going to allcate
 * 
 * \return  return the number of byte of the virtial address
 * 
 * \endcond
 * ================================================================================================
 */
unsigned long long 
MMU::lookup (int va)
{
    pair<Page*, unsigned long long> dummy;
    if (mTLB.lookup(va, dummy)) return dummy.second;
    
    return 0;
}


/** ===============================================================================================
 * \name    memoryAllocate
 * 
 * \brief   Allocate memory for CPU virtual address to physical address.
 * 
 * \param   va          the virtual address going to allcate
 * \param   numOfByte   the number of Bytes needs to allocate
 * 
 * \endcond
 * ================================================================================================
 */
bool 
MMU::memoryAllocate (int va, unsigned long long numOfByte)
{
    ASSERT(numOfByte > 0, "Try to allocate zero space");
    
    if (lookup(va)) 
    {
        log_I("memoryAllocate", "VA: " + to_string(va) + " Size: " + to_string(numOfByte) + " The virtual address already been allocated");
        return false;
    }

#if (PRINT_MEMORY_ALLOCATION)
    log("memoryAllocate", "VA: " + to_string(va) + " Size: " + to_string(numOfByte), Color::Cyan);
#endif

    Page* PP = mMC->memoryAllocate(numOfByte);
    mTLB.insert(va, make_pair(PP, numOfByte));

    return true;
}


/** ===============================================================================================
 * \name    memoryRelease
 * 
 * \brief   Release memory for CPU virtual address to physical address.
 * 
 * \param   va      the virtual address going to allcate
 * 
 * \note    all the pages used in this va will be release, note the casecode layers
 * 
 * \endcond
 * ================================================================================================
 */
void 
MMU::memoryRelease (int va)
{
    pair<Page*, unsigned long long> pa_pair;
    if (!mTLB.lookup(va, pa_pair)) return;

    mMC->memoryRelease(pa_pair.first);
    mTLB.erase(va);
}


/** ===============================================================================================
 * \name    addressTranslate
 * 
 * \brief   Translate the virtual address to physical address.
 * 
 * \param   va      the virtual address going to translate
 * 
 * \return  Pythsical Pages vector
 * 
 * \endcond
 * ================================================================================================
 */
vector<unsigned long long>
MMU::addressTranslate (int va)
{
    /* lookup wheather VA has been cached */
    log_V("addressTranslate", to_string(va));

    pair<Page*, unsigned long long> pa_pair;
    ASSERT(mTLB.lookup(va, pa_pair), "The virtual address haven't been allocated");

    vector<unsigned long long> pa_list;
    Page* page = pa_pair.first;

    while(page != nullptr) 
    {
        pa_list.emplace_back(page->pageIndex);
        page = page->nextPage;
    }

    return move(pa_list);
}


/** ===============================================================================================
 * \name    getPageSummary
 * 
 * \brief   Summary the page information used in specific virtual address.
 * 
 * \param   va      the virtual address
 * 
 * \return  the page information
 * 
 * \note    the recorded information will be reset after summary
 * 
 * \endcond
 * ================================================================================================
 */
PageRecord 
MMU::getPageSummary (int va)
{
    pair<Page*, unsigned long long> pa_pair;
    if (!mTLB.lookup(va, pa_pair)) return {};

    /* Perform release */
    PageRecord record;
    Page* page = pa_pair.first;
    while(page)
    {
        record += page->record;
        page->record = PageRecord();
        page = page->nextPage;
    }
    
    return record;

}