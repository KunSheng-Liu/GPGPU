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
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include <string.h>
#include <sys/time.h>

/* ************************************************************************************************
 * Name Space
 * ************************************************************************************************
 */
using namespace std;


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

/* Peripheral */

/* Approach  */


/* ************************************************************************************************
 * Application Configuration
 * ************************************************************************************************
 */
#define LOG_LEVEL               DEBUG
#define PRINT_MODEL_DETIAL      true


#endif