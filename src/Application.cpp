/**
 * \name    Application.cpp
 * 
 * \brief   Implement the API
 * 
 * \date    APR 16, 2023
 */

#include "include/Application.hpp"

/* ************************************************************************************************
 * Global Variable
 * ************************************************************************************************
 */
int Application::appCount = 0;


/** ===============================================================================================
 * \name    Application
 * 
 * \brief   Construct a model
 * 
 * \param   model_type    the model type
 * 
 * \endcond
 * ================================================================================================
 */
Application::Application(char* model_type): appID(++appCount)
{
    mModel = new Model();
    if (strcmp(model_type, "None") == 0) {

        mModel->None();

    } else if (strcmp(model_type, "VGG16") == 0) {

        mModel->VGG16();

    } else if (strcmp(model_type, "ResNet18") == 0) {

        mModel->ResNet18();

    }
    
#if PRINT_MODEL_DETIAL
    mModel->printSummary();
#endif

}


/** ===============================================================================================
 * \name   ~Application
 * 
 * \brief   Destruct a model
 * 
 * \param   name    the model name
 * 
 * \endcond
 * ================================================================================================
 */
Application::~Application()
{
    delete mModel;
}