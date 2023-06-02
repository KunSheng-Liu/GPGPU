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
typedef enum {
    Default = 0,
    Red     = 31,
    Green   = 32,
    Yellow  = 33,
    Blue    = 34,
    Cyan    = 36,
} Color;

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */

inline void log (std::string tag, std::string logInfo, Color color = Color::Default)
{
    std::cout << "\033[1;" << color << "m" << tag << ": " << logInfo << "\033[0m" << std::endl;
}

/* ERROR */
inline void log_E (std::string tag, std::string logInfo)
{
#if LOG_LEVEL >= ERROR
    log (tag, logInfo, Color::Red);
#endif
}

/* WARNNING */
inline void log_W (std::string tag, std::string logInfo)
{
#if LOG_LEVEL >= WARNNING
    log (tag, logInfo, Color::Blue);
#endif
}

/* INFO */
inline void log_I (std::string tag, std::string logInfo)
{
#if LOG_LEVEL >= INFO
    log (tag, logInfo, Color::Green);
#endif
}

/* DEBUG */
inline void log_D (std::string tag, std::string logInfo)
{
#if LOG_LEVEL >= DEBUG
    log (tag, logInfo, Color::Cyan);
#endif
}

/* TRACE */
inline void log_T (std::string tag, std::string logInfo)
{
#if LOG_LEVEL >= TRACE
    log (tag, logInfo, Color::Yellow);
#endif
}

/* VERBOSE */
inline void log_V (std::string tag, std::string logInfo)
{
#if LOG_LEVEL >= VERBOSE
    log (tag, logInfo, Color::Default);
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