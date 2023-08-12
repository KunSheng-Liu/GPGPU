/**
 * \name    LayerGroup.hpp
 * 
 * \brief   Use to contain a sequential of layer
 *          
 * \date    Apr 1, 2023
 */

#ifndef _LAYERGROUP_HPP_
#define _LAYERGROUP_HPP_

/* ************************************************************************************************
 * Include Library
 * ************************************************************************************************
 */
#include "App_config.h"
#include "Log.h"

#include "Layers.hpp"


/* ************************************************************************************************
 * Enumeration
 * ************************************************************************************************
 */

/* The type of this group */
enum class Group_t{
    CaseCade,
    CaseCode,
};


/** ===============================================================================================
 * \name    LayerGroup
 * 
 * \brief   The layer container to keep a sequential layer. You can add the layer or layerGroup into 
 *          this container.
 * \endcond
 * ================================================================================================
 */
class LayerGroup: public Layer
{
/* ************************************************************************************************
 * Class Constructor
 * ************************************************************************************************
 */ 
public:

    LayerGroup(Group_t = Group_t::CaseCade, char* = (char*)"LayerGroup");

   ~LayerGroup();

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    void addLayer (Layer*);
    void setIFMap  (pair<int, vector<DATA_TYPE>*> data) override;
    void setOFMap  (pair<int, vector<DATA_TYPE>*> data) override;
    void setFilter (pair<int, vector<DATA_TYPE>*> data) override;

    void printInfo() override;
    void changeBatch (int new_batch_size) override;
    void memoryAllocate (MMU* mmu) override;
    vector<Kernel*> compileToKernel(int app_id, int model_id, vector<Kernel>& container, vector<Kernel*> dependency) override;
    void issueLayer(ThreadArg* threadArg) override {}
    
private:
    void addCaseCade (Layer*);
    void addCaseCode (Layer*);

    void calculateOFMapSize() override {};


/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
public:
    const Group_t groupType;

    /* The container of the layers */
    vector<Layer*> layers;
};



/** ===============================================================================================
 * \name    ResNetBlock18
 * 
 * \brief   The layergroup prototype used in ResNet18
 * 
 * \endcond
 * ================================================================================================
 */
class ResNetBlock18: public LayerGroup
{
/* ************************************************************************************************
 * Class Constructor
 * ************************************************************************************************
 */ 
public:
    ResNetBlock18(int& layer_id, vector<int>, bool = false);

private:
/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
    void BasicBlock(int& layer_id, vector<int>);
    void BottleNeckBlock(int& layer_id, vector<int>);
};



/** ===============================================================================================
 * \name    Inception
 * 
 * \brief   The layergroup prototype used in GoogleNet
 * 
 * \endcond
 * ================================================================================================
 */
class Inception: public LayerGroup
{
/* ************************************************************************************************
 * Class Constructor
 * ************************************************************************************************
 */ 
public:

    Inception(int& layer_id, vector<int>, int, int, int, int, int, int);
    
/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
private:
    int channel_1x1;
    int channel_reduce_3x3;
    int channel_3x3;
    int channel_reduce_5x5;
    int channel_5x5;
    int channel_pooling;
};



/** ===============================================================================================
 * \name    Fire
 *
 * \brief   The layergroup prototype used in SqueezeNet
 * 
 * \endcond
 * ================================================================================================
 */
class Fire: public LayerGroup
{
/* ************************************************************************************************
 * Class Constructor
 * ************************************************************************************************
 */ 
public:

    Fire(int& layer_id, vector<int>, int, int, int);
    
/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
private:
    int channel_s1x1;
    int channel_e1x1;
    int channel_e3x3;
};

#endif