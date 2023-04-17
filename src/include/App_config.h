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
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <queue>
#include <unordered_map>
#include <vector>

#include <string.h>
#include <sys/time.h>

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
class TLB;
class Application;
class Model;
class InferenceEngine;
class MemoryControl;


/* ************************************************************************************************
 * Enumeration
 * ************************************************************************************************
 */
/* Logger level */
#define ERROR               0
#define WARNNING            1
#define INFO                2
#define DEBUG               3
#define VERBOSE             4

/* Approach  */


/* ************************************************************************************************
 * Application Configuration
 * ************************************************************************************************
 */
#define LOG_LEVEL               DEBUG
#define PRINT_MODEL_DETIAL      true


/* ************************************************************************************************
 * BenchMark
 * ************************************************************************************************
 */
#define HARDWARE_ARCHITECTURE   AGX_XAVIER

/* ************************************************************************************************
 * Hardware Configuration
 * ************************************************************************************************
 */
#if (HARDWARE_ARCHITECTURE == AGX_XAVIER)
    /* Architecture */
    #define PAGE_SIZE                   4096                    // unit (Byte)
    #define DRAM_SPACE                  1  * pow(2, 28)         // unit (Byte)  256 MB
    #define DISK_SPACE                  16 * pow(2, 30)         // unit (Byte)   16 GB
    #define PCIE_CHANNEL                16
    #define PCIE_Bendwidth              1  * pow(10, 9)         // unit (s)
    #define PAGE_FAULT_PENALTY          50 * pow(0.1, 6)		// unit (s)

    #define GPU_PREFETCH_SIZE           80

    /* Frequency */
    #define CPU_F                       1200000000.0            // unit (Hz)    1200 MHz
    #define MC_F                        4266000000.0            // unit (Hz)    4266 MHz
    #define GPU_F                       1377000000.0            // unit (Hz)    1377 MHz
    #define GMMU_F                      1377000000.0            // unit (Hz)    1377 MHz
    
    /* CPU */
    #define CPU_CONSTANT_POWER	        1526 * pow(0.1, 3)      // unit (W) (8 core, Frequence=1200000000)

    /* DRAM */
    #define DRAM_READ_LATENCY	        10   * pow(0.1, 10)	    // unit (s) about 30GB/s
    #define DRAM_WRITE_LATENCY	        10   * pow(0.1, 10)	    // unit (s) about 30GB/s
    #define DRAM_READ_ENENGY	        2.3  * pow(0.1, 9)	    // unit (J)
    #define DRAM_WRITE_ENENGY	        2.44 * pow(0.1, 9)	    // unit (J)
    #define DRAM_LEAKAGE_POWER	        70.8 * pow(0.1, 3)	    // unit (W)

    /* GPU */
    #define GPU_SM_NUM                  8 
    #define GPU_WARP_PER_SM             2 
    #define GPU_THREAD_PER_WARP         32 
    #define GPU_MAX_THREAD_PER_SM       GPU_SM_NUM *  GPU_WARP_PER_SM * GPU_THREAD_PER_WARP
    #define GPU_REGISTER_PER_SM         65536 

    #define GPU_SHARED_MEMORY_PER_SM    96  	                // unit (KB)
    #define GPU_L1_CACHE                192                     // unit (KB)
    #define GPU_L2_CACHE                512                     // unit (KB)
    #define GPU_GDDR_SIZE               

    #define GPU_IDEL_POWER		        10    * pow(0.1, 3)      // unit (W) (Frequence=1377000000)
    #define GPU_EXEC_POWER		        19326 * pow(0.1, 3)      // unit (W) (Frequence=1377000000)
#endif

#endif