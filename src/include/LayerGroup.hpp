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

    LayerGroup(Group_t = Group_t::CaseCade);

   ~LayerGroup();

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    void addLayer (Layer*);
    void setIFMap  (vector<unsigned char>* data) override;
    void setFilter (vector<unsigned char>* data) override;

    void printInfo() override;
    void changeBatch (int new_batch_size) override;
    void memoryAllocate (MMU* mmu) override;
    vector<Kernel*> issueLayer(vector<Kernel>& container, vector<Kernel*> dependency) override;
    
private:
    void addCaseCade (Layer*);
    void addCaseCode (Layer*);

    void calculateOFMapSize() override;


/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
public:
    const Group_t groupType;

    /* The container of the layers */
    vector<Layer*> layers;
};


#endif