/** 
 * \name    Log.hpp
 * 
 * \brief   Log the information of the application
 * 
 * \date    Mar 31, 2023
 */

#ifndef _LOG_HPP_
#define _LOG_HPP_

/* ************************************************************************************************
 * Include Library
 * ************************************************************************************************
 */
#include "App_config.hpp"

#include <iostream>

/* ************************************************************************************************
 * Enumeration
 * ************************************************************************************************
 */
typedef enum {
    
}Log_t;

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
void log (std::string tag, Log_t logType);

/* ERROR */
inline void log_E (std::string tag, std::string logInfo)
{
#if LOG_LEVEL >= ERROR
    std::cout << "\033[1;31mLogE:\033[0m Tag: " << tag << ": " << logInfo << std::endl;
#endif
}

/* WARNNING */
inline void log_W (std::string tag, std::string logInfo)
{
#if LOG_LEVEL >= WARNNING
    std::cout << "\033[1;34mLogW:\033[0m Tag: " << tag << ": " << logInfo << std::endl;
#endif
}

/* INFO */
inline void log_I (std::string tag, std::string logInfo)
{
#if LOG_LEVEL >= INFO
    std::cout << "\033[1;32mLogI:\033[0m Tag: " << tag << ": " << logInfo << std::endl;
#endif
}

/* DEBUG */
inline void log_D (std::string tag, std::string logInfo)
{
#if LOG_LEVEL >= DEBUG
    std::cout << "\033[1;36mLogD:\033[0m Tag: " << tag << ": " << logInfo << std::endl;
#endif
}

/* VERBOSE */
inline void log_V (std::string tag, std::string logInfo)
{
#if LOG_LEVEL >= VERBOSE
    std::cout << "LogV: Tag: " << tag << ": " << logInfo << std::endl;
#endif
}

#endif