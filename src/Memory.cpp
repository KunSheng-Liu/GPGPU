/**
 * \name    Memory.cpp
 * 
 * \brief   Implement the API
 * 
 * \date    APR 4, 2023
 */

#include "include/Memory.hpp"


/** ===============================================================================================
 * \name    Memory
 * 
 * \brief   The base class of the memory hierarchy. You can implement the cache, RAM by inheritance 
 *          this class and provide the following parameter.
 * 
 * \param   
 * 
 * \endcond
 * ================================================================================================
 */
Memory::Memory(int size) : storageSize(size)
{
    data.reserve(size);
}




