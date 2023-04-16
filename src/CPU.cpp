/**
 * \name    CPU.cpp
 * 
 * \brief   Implement the CPU and it's components.
 * 
 * \date    APR 6, 2023
 */

#include "include/CPU.hpp"

/** ===============================================================================================
 * \name    CPU
 * 
 * \brief   The class of the CPU, contains MMU, TLB.
 * 
 * \param   mc      the pointer of memory controller
 * 
 * \endcond
 * ================================================================================================
 */
CPU::CPU(MemoryControl* mc) : mMC(mc)
{
    mMMU = new MMU(mc);

    APPs.push_back(new Application ((char*)"VGG16"));
    APPs.push_back(new Application ((char*)"ResNet18"));
    
    for (auto app: APPs)
    {
        app->mModel->setBatchSize(1);
        app->mModel->memoryAllocate(mMMU);
    }
    
}


/** ===============================================================================================
 * \name    CPU
 * 
 * \brief   The class of the CPU, contains MMU, TLB.
 * 
 * \param   mc      the pointer of memory controller
 * 
 * \endcond
 * ================================================================================================
 */
CPU::~CPU()
{

}
