/** 
 * \name    APP_config.h
 * 
 * \brief   Declare all configure
 * 
 * \date    Mar 31, 2023
 */

#ifndef _APP_CONFIG_H_
#define _APP_CONFIG_H_

/* ************************************************************************************************
 * Include Library
 * ************************************************************************************************
 */
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <list>
#include <queue>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <unistd.h>
#include <pthread.h>
#include <string.h>
#include <sys/time.h>

#include "Global.h"
#include "Macro.h"

/* ************************************************************************************************
 * Name Space
 * ************************************************************************************************
 */
using namespace std;

/* ************************************************************************************************
 * Pre-declaration Class
 * ************************************************************************************************
 */
class GPGPU;

class CPU;
class MMU;
class Scheduler;

class Application;
class Model;
class Layer;
class LayerGroup;

class MemoryController;

class GPU;
class GMMU;
class SM;
class Warp;
class Block;
class Kernel;
class Request;


/* ************************************************************************************************
 * Enumeration
 * ************************************************************************************************
 */

/* ************************************************************************************************
 * Print-Out Configuration
 * ************************************************************************************************
 */
#define LOG_LEVEL                           WARNNING
#define LOG_OUT_PATH                        "./log/"
#define PRINT_TIME_STEP                     false
#define PRINT_SM_ALLCOATION_RESULT          false
#define PRINT_MODEL_DETIAL                  false
#define PRINT_MEMORY_ALLOCATION             false
#define PRINT_ACCESS_PATTERN                false
#define PRINT_BLOCK_RECORD                  true
#define PRINT_WARP_RECORD                   true

#define PRINT_LAZY_BATCHING                 true


/* ************************************************************************************************
 * BenchMark
 * ************************************************************************************************
 */
#define HARDWARE_ARCHITECTURE               AGX_XAVIER

/* ************************************************************************************************
 * Software Configuration
 * ************************************************************************************************
 */ 
#define SIMULATION_TIME                     100                     // unit (ms)
#define TASK_DEADLINE                       100                     // unit (ms)
#define PERFECT_ACCESS                      false                   // Is VRAM space is always enough ?
#define COMPULSORY_MISS                     true                    // Initial the data in the DRAM ?
#define THREAD_KERNEL_COMPILE               true
#define THREAD_NUM                          8

/* ************************************************************************************************
 * Hardware Configuration
 * ************************************************************************************************
 */
#if (HARDWARE_ARCHITECTURE == AGX_XAVIER)
    /* Architecture */
    #define PAGE_SIZE                       4096                    // unit (Byte)
    #define DRAM_SPACE                      32    * pow(2, 30)      // unit (Byte)   32 GB
    #define DISK_SPACE                      32    * pow(2, 30)      // unit (Byte)   32 GB
    #define PRE_ALLOCATE_SIZE               256   * pow(2, 20)      // unit (Byte)  256 MB
    #define PCIE_CHANNEL                    16
    #define PCIE_BANDWIDTH                  16    * pow(10, 9)      // unit (B/s)  16 GB/s
    #define PAGE_FAULT_PENALTY              20    * pow(0.1, 6)     // unit (s)

    #define PCIE_ACCESS_BOUND               80

    /* Frequency */ 
    #define CPU_F                           1200000000.0            // unit (Hz)    1200 MHz
    #define MC_F                            4266000000.0            // unit (Hz)    4266 MHz
    #define GPU_F                           1377000000.0            // unit (Hz)    1377 MHz
    #define GMMU_F                          1377000000.0            // unit (Hz)    1377 MHz

    /* CPU */   
    #define CPU_CONSTANT_POWER	            1526  * pow(0.1, 3)     // unit (W) (8 core, Frequence=1200000000)

    /* DRAM */  
    #define DRAM_READ_LATENCY	            10    * pow(0.1, 10)	// unit (s) about 30GB/s
    #define DRAM_WRITE_LATENCY	            10    * pow(0.1, 10)	// unit (s) about 30GB/s
    #define DRAM_READ_ENENGY	            2.3   * pow(0.1, 9)	    // unit (J)
    #define DRAM_WRITE_ENENGY	            2.44  * pow(0.1, 9)	    // unit (J)
    #define DRAM_LEAKAGE_POWER	            70.8  * pow(0.1, 3)	    // unit (W)

    /* GPU */   
    #define GPU_SM_NUM                      8 
    #define GPU_MAX_WARP_PER_SM             64 
    #define GPU_MAX_WARP_PER_BLOCK          32 
    #define GPU_MAX_THREAD_PER_WARP         32 
    #define GPU_MAX_THREAD_PER_SM           GPU_MAX_THREAD_PER_WARP * GPU_MAX_THREAD_PER_WARP
    #define GPU_MAX_BLOCK_PER_SM            32
    #define GPU_MAX_THREAD_PER_BLOCK        1024
    #define GPU_MAX_ACCESS_NUMBER           32 
    #define GPU_REGISTER_PER_SM             65536 

    #define GPU_SHARED_MEMORY_PER_SM        96    * pow(2, 10) 	    // unit (KB)
    #define GPU_L1_CACHE_SIZE               48    * 1024            // unit (B)
    #define GPU_L1_CACHE_LINE_SIZE          32                      // unit (B)
    #define GPU_L1_CACHE_WAY_NUMBER         4                       // unit (B)
    #define GPU_L1_CACHE_BLOCK_SIZE         GPU_L1_CACHE_LINE_SIZE * GPU_L1_CACHE_WAY_NUMBER  // unit (B)

    #define GPU_L2_CACHE_SIZE               4     * 1024 * 1024     // unit (B)
    #define GPU_L2_CACHE_LINE_SIZE          32                      // unit (B)
    #define GPU_L2_CACHE_WAY_NUMBER         16                      // unit (B)
    #define GPU_L2_CACHE_BLOCK_SIZE         GPU_L1_CACHE_LINE_SIZE * GPU_L1_CACHE_WAY_NUMBER  // unit (B)
    #define GPU_VRAM_SIZE                   

    #define GPU_IDEL_POWER		            10    * pow(0.1, 3)     // unit (W) (Frequence=1377000000)
    #define GPU_EXEC_POWER		            19326 * pow(0.1, 3)     // unit (W) (Frequence=1377000000)
#endif


/* ************************************************************************************************
 * Global Configuration
 * ************************************************************************************************
 */
#define PAGE_FAULT_COMMUNICATION_CYCLE      PAGE_FAULT_PENALTY * GMMU_F     // unit (cycle)
#define PAGE_FAULT_MIGRATION_UNIT_CYCLE     ceil((PAGE_SIZE) / (PCIE_BANDWIDTH) * (GMMU_F))  // unit (cycle)
#define TRANSFER_SET_SIZE                   80 

/* ************************************************************************************************
 * Other Flags
 * ************************************************************************************************
 */
#define RECORD_MODEL_INFORMATIONS           false


#endif