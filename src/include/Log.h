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

/* TRACE */
inline void log_T (std::string tag, std::string logInfo)
{
#if LOG_LEVEL >= TRACE
    std::cout << "\033[1;33m" << tag << ": " << logInfo << "\033[0m" << std::endl;
#endif
}

/* VERBOSE */
inline void log_V (std::string tag, std::string logInfo)
{
#if LOG_LEVEL >= VERBOSE
    std::cout << tag << ": " << logInfo << std::endl;
#endif
}

/* File */
inline void log_to_file (std::ofstream file, std::string logInfo){
    if (file.is_open()) {
        file << logInfo << std::endl;
    } else {
        log_E("log_to_file", "file is not open");
    }
}

#endif