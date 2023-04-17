/**
 * \name    InferenceEngine.hpp
 * 
 * \brief   Declare the gpu driver API 
 * 
 * \date    APR 17, 2023
 */

#ifndef _INFERENCE_ENGINE_HPP_
#define _INFERENCE_ENGINE_HPP_

/* ************************************************************************************************
 * Include Library
 * ************************************************************************************************
 */
#include "App_config.h"
#include "Log.h"

#include "CPU.hpp"
#include "Application.hpp"
#include "Models.hpp"

/** ===============================================================================================
 * \name    InferenceEngine
 * 
 * \brief   The engine contain the inference scheduler and the gpu driver.
 * 
 * \endcond
 * ================================================================================================
 */
class InferenceEngine
{
/* ************************************************************************************************
 * Class Constructor
 * ************************************************************************************************
 */ 
public:

    InferenceEngine(MMU* mmu, vector<Application*>* apps);

    ~InferenceEngine();


/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    void cycle ();

    void Dynamic_Batching_Algorithm ();
    void Kernek_Inference_Scheduler ();

/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
private:

    MMU* mMMU;

    vector<Application*>* mAPPs;
};

#endif