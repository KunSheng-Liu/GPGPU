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
CPU::CPU(MemoryController* mc) : mMC(mc)
{
    mMMU = new MMU(mc);
    mInferenceEngine = new InferenceEngine(mMMU, &mAPPs);

    mAPPs.push_back(new Application ((char*)"VGG16"));
    mAPPs.push_back(new Application ((char*)"ResNet18"));
    
}


/** ===============================================================================================
 * \name    CPU
 * 
 * \brief   Destruct CPU
 * 
 * \endcond
 * ================================================================================================
 */
CPU::~CPU()
{

}


/** ===============================================================================================
 * \name    cycle
 * 
 * \brief   Handling the CPU in a period of cycle
 * 
 * \endcond
 * ================================================================================================
 */
void
CPU::cycle()
{
    for (auto app: mAPPs)
    {
        app->cycle();
    }

    mInferenceEngine->cycle();
}
