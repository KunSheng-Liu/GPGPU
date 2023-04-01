/** 
 * \name    Log.h
 * 
 * \brief   Log the information of the application
 * 
 * \date    Mar 31, 2023
 */

#ifndef _LOG_H_
#define _LOG_H_

/* ************************************************************************************************
 * Include Library
 * ************************************************************************************************
 */
#include "App_config.h"

/* ************************************************************************************************
 * Enumeration
 * ************************************************************************************************
 */


/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
/* ERROR */
inline void log_E (std::string tag, std::string logInfo)
{
#if LOG_LEVEL >= ERROR
    std::cout << "\033[1;31m" << tag << ": " << logInfo << "\033[0m" << std::endl;
#endif
}

/* WARNNING */
inline void log_W (std::string tag, std::string logInfo)
{
#if LOG_LEVEL >= WARNNING
    std::cout << "\033[1;34m" << tag << ": " << logInfo << "\033[0m" << std::endl;
#endif
}

/* INFO */
inline void log_I (std::string tag, std::string logInfo)
{
#if LOG_LEVEL >= INFO
    std::cout << "\033[1;32m" << tag << ": " << logInfo << "\033[0m" << std::endl;
#endif
}

/* DEBUG */
inline void log_D (std::string tag, std::string logInfo)
{
#if LOG_LEVEL >= DEBUG
    std::cout << "\033[1;36m" << tag << ": " << logInfo << "\033[0m" << std::endl;
#endif
}

/* VERBOSE */
inline void log_V (std::string tag, std::string logInfo)
{
#if LOG_LEVEL >= VERBOSE
    std::cout << tag << ": " << logInfo << std::endl;
#endif
}

/* ************************************************************************************************
 * Marco
 * ************************************************************************************************
 */
#define GET_FUNCTION( _1, _2, function, ... ) function 
/* Exit when condition == false */
#define ASSERT( ... )   GET_FUNCTION( __VA_ARGS__, ASSERT_2, ASSERT_1 ) (__VA_ARGS__)
// #define DEBUG( ... )    GET_FUNCTION( __VA_ARGS__, DEBUG_2, DEBUG_1 )   (__VA_ARGS__)

/* Assert function with print message */
#define ASSERT_1( condition ) { if (!(condition)) exit (1); }
#define ASSERT_2( condition, ... ) {                                           \
    if (!(condition)) {                                                        \
        std::cout << __FILE__  << ": " << __LINE__ << ": " <<  __func__        \
                  << ": " << __VA_ARGS__ << std::endl;                         \
    exit (1);} }

#endif