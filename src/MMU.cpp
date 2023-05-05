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
MMU::MMU(MemoryController* mc): mMC(mc), mTLB(TLB(DRAM_SPACE / PAGE_SIZE))
{

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
void 
MMU::memoryAllocate (int va, int numOfByte)
{
    if (va == 0 || numOfByte == 0) return;
    
    pair<Page*, int> pa_pair = mTLB.lookup(va);
    if (pa_pair.first == nullptr && pa_pair.second == -1) 
    {
        log_D("memoryAllocate", "VA: " + to_string(va) + " Size: " + to_string(numOfByte));
        Page* PP = mMC->memoryAllocate(numOfByte);
        mTLB.insert(va, make_pair(PP, numOfByte));
    }
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
vector<int>
MMU::addressTranslate (int va)
{
    /* lookup wheather VA has been cached */
    pair<Page*, int> pa_pair = mTLB.lookup(va);
    
    ASSERT(!(pa_pair.first == nullptr || pa_pair.second == -1), "The virtual address haven't been allocated");

    vector<int> pa_list;
    Page* page = pa_pair.first;
    int pageIndex = page->pageIndex;

    while(page != nullptr) 
    {
        pa_list.emplace_back(page->pageIndex);
        page = page->nextPage;
    }

    return move(pa_list);
}



/** ===============================================================================================
 * \name    lookup
 * 
 * \brief   lookup PA pair by VA
 * 
 * \param   va      the virtual address
 * 
 * \return  a pair of physical page and the data size
 * 
 * \endcond
 * ================================================================================================
 */
pair<Page*, int>
TLB::lookup (int va) {

    auto it = table.find(va);

    if (it != table.end()) // TLB hit
    {
        list.splice(list.begin(), list, it->second);
        return it->second->PAPair;

    } else { // TLB miss

        return make_pair(nullptr, -1);
    }
}


/** ===============================================================================================
 * \name    insert
 * 
 * \brief   insert the VA and PA to the table
 * 
 * \param   va          the virtual address
 * \param   pa_pair     the pair of physical page and the data size
 * 
 * \endcond
 * ================================================================================================
 */
void 
TLB::insert(int va, pair<Page*, int> pa_pair) {

    auto it = table.find(va);

    if (it != table.end()) // Page exist
    {
        it->second->PAPair = pa_pair;
        list.splice(list.begin(), list, it->second);

    } else {  // Page not exist
        
        /* TLB already full */
        if (table.size() >= capacity) {
            int vpn = list.back().VA;
            table.erase(vpn);
            list.pop_back();
        }

        /* Add new entry */
        list.emplace_front(va, pa_pair);
        table[va] = list.begin();
    }
}