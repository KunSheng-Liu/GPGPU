/** 
 * \name    Approach.hpp
 * 
 * \brief   Declare the approach function pointer
 * 
 * \date    May 23, 2023
 */
#ifndef _APPROACH_HPP_
#define _APPROACH_HPP_

#include "App_config.h"
#include "CPU.hpp"
#include "Kernel.hpp"

/* ************************************************************************************************
 * Callback Functions
 * ************************************************************************************************
 */
#ifndef _BASELINE_
    bool Greedy_Inference_Admission   ( CPU* cpu );
    bool Baseline_Inference_Admission ( CPU* cpu );

    bool Default_Model_Launcher       ( CPU* cpu );

    bool Baseline_Kernel_Scheduler    ( CPU* cpu );

#endif

#ifndef _BARM_
    bool BARM_Inference_Admission ( CPU* cpu );
#endif

#ifndef _LAZY_BATCHING_
    bool Lazy_Batching_Kernel_Scheduler ( CPU* cpu );
#endif

#ifndef _MY_APPROACH_
    bool Dynamic_Batch_Admission    ( CPU* cpu );
    bool Kernel_Inference_Scheduler ( CPU* cpu );
#endif

#endif