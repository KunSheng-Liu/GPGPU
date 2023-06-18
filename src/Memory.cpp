/**
 * \name    Memory.cpp
 * 
 * \brief   Implement the API
 * 
 * \date    APR 4, 2023
 */

#include "include/Memory.hpp"

/* ************************************************************************************************
 * Global Variable
 * ************************************************************************************************
 */
int Memory::memoryCount  = 0;
unsigned long long Memory::storageCount = PAGE_SIZE;    // the first page in not allow to use

/** ===============================================================================================
 * \name    Memory
 * 
 * \brief   The base class of the memory hierarchy. You can implement the RAM, Disk by inheritance 
 *          this class and provide the following parameter.
 * 
 * \param   memory_type         the memory type
 * \param   storage_size        the memory storage size.                unit in (Byte).
 * \param   total_bandwidth     the total IO bandwidth of the memory.   unit in (Bits).
 * \param   channel_bandwidth   a sigal channel's bandwidth.            unit in (Bits).
 * 
 * \endcond
 * ================================================================================================
 */
Memory::Memory(Memory_t memory_type, unsigned long long storage_size, int total_bandwidth, int channel_bandwidth) 
    : memoryIndex(memoryCount++), memoryType(memory_type), recorder(MemoryRecord())
    , storageSize(storage_size), startPhysicalAddress(storageCount), endPhysicalAddress(storageCount + storage_size)
    , totalBandwidth(total_bandwidth), channelBandwidth(channel_bandwidth)
{
    /* Create physical storage */
    ASSERT(storage_size % 4 == 0, "Error storage size, should be align in 4 Byte");
    // for (int i = startPhysicalAddress; i < endPhysicalAddress; i += 4) storage.insert({i, -1});

    for (int i = 0; i < totalBandwidth / channelBandwidth; i++)
    {
        IO_Channels.insert({i, IO_Channel()});
        idleChannelList.emplace_back(i);
    }

    access_finish_queue  = {};

    storageCount += storageSize;
}


/** ===============================================================================================
 * \name    Memory
 * 
 * \brief   Destruct Memory
 * 
 * \endcond
 * ================================================================================================
 */
Memory::~Memory()
{
    for (auto channel : IO_Channels) ASSERT(channel.second.access == nullptr, "destruct error");
    
    ASSERT(access_finish_queue.empty(),  "destruct error");
}


/** ===============================================================================================
 * \name    cycle
 * 
 * \brief   Handling the memory read/write access in a period of cycle
 * 
 * \endcond
 * ================================================================================================
 */
void
Memory::cycle()
{
    log_T("Memory Cycle", to_string(memoryType));

    bool exec = false;
    for (auto channel : IO_Channels)
    {
        if (channel.second.access && (exec = true) &&--channel.second.waiting_cycle == 0)
        {
            access_finish_queue.insert(move(channel.second.access));
            idleChannelList.push_back(channel.first);
        }
    }

    exec ? recorder.exec_cycle++ : recorder.idle_cycle++;
}


/** ===============================================================================================
 * \name    Read
 * 
 * \brief   Launch a read access to memory
 * 
 * \param   num_of_bytes    number of bytes read from this memory
 * \param   access          the source access going to read
 * 
 * \endcond
 * ================================================================================================
 */
bool
Memory::Read(int num_of_bytes, MemoryAccess* access)
{
    if (!idleChannelList.empty())
    {
        int channel_id = idleChannelList.front();

        ASSERT(IO_Channels[channel_id].access == nullptr, "idelChannel exist access");
        IO_Channels[channel_id].access = access;
        IO_Channels[channel_id].waiting_cycle = min (num_of_bytes / (channelBandwidth / 8), (unsigned) 1);

        idleChannelList.pop_front();

        return true;
    }
    
    return false;
}


/** ===============================================================================================
 * \name    Write
 * 
 * \brief   Launch a write access to memory
 * 
 * \param   num_of_bytes    number of bytes write to this memory
 * \param   access          the source access going to write
 * 
 * \endcond
 * ================================================================================================
 */
bool
Memory::Write(int num_of_bytes, MemoryAccess* access)
{
    if (!idleChannelList.empty())
    {
        int channel_id = idleChannelList.front();

        ASSERT(IO_Channels[channel_id].access == nullptr, "idelChannel exist access");
        IO_Channels[channel_id].access = access;
        IO_Channels[channel_id].waiting_cycle = min (num_of_bytes / (channelBandwidth / 8), (unsigned) 1);

        idleChannelList.pop_front();

        return true;
    }
    
    return false;
}


/** ===============================================================================================
 * \name    DRAM
 * 
 * \brief   Construct a DDR type memory
 * 
 * \param   storage_size    the memory storage size. unit in (Byte).
 * 
 * \endcond
 * ================================================================================================
 */
DRAM::DRAM(unsigned long long storage_size) : Memory(Memory_t::SPACE_DRAM, storage_size, DRAM_TOTAL_BANDWIDTH, DRAM_CAHNNEL_BANDWIDTH)
{
    
}
