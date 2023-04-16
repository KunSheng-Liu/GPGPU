/**
 * \name    Application.hpp
 * 
 * \brief   Declare the structure of Application
 * 
 * \note    Contains the model and it's data
 * 
 * \date    APR 16, 2023
 */

#ifndef _APPLICATION_HPP_
#define _APPLICATION_HPP_

/* ************************************************************************************************
 * Include Library
 * ************************************************************************************************
 */
#include "App_config.h"
#include "Log.h"
#include "Models.hpp"

/** ===============================================================================================
 * \name    Application
 * 
 * \brief   Contains the model and it's data
 * 
 * \endcond
 * ================================================================================================
 */
class Application
{
/* ************************************************************************************************
 * Class Constructor
 * ************************************************************************************************
 */ 
public:

    Application(char*);

   ~Application();

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    vector<int>* getIFMapSize  (void) {return mModel->getIFMapSize();}
    vector<int>* getOFMapSize  (void) {return mModel->getOFMapSize();}

/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
public:
    const int appID;

private:

    /* Number of layer be created */
    static int appCount;

public:
    Model* mModel;
    vector<vector<unsigned char>> inputDatas;
};

#endif