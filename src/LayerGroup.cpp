/**
 * \name    Layer.cpp
 * 
 * \brief   Implement the API
 * 
 * \date    Mar 31, 2023
 */

#include "include/LayerGroup.hpp"

/** ===============================================================================================
 * \name    LayerGroup
 *
 * \brief   Construct a layerGroup
 * 
 * \endcond
 * ================================================================================================
 */
LayerGroup::LayerGroup(Group_t group_type): Layer(), groupType(group_type)
{
    /* Group not in count of a layer */
    layerCount--;
}


/** ===============================================================================================
 * \name   ~LayerGroup
 *
 * \brief   Construct a layerGroup
 * 
 * \endcond
 * ================================================================================================
 */
LayerGroup::~LayerGroup()
{
    // delete &layers;
}


/** ===============================================================================================
 * \name    addLayer
 *
 * \brief   Add layer into the layer graph.
 * 
 * \endcond
 * ================================================================================================
 */
void::
LayerGroup::addLayer (Layer* layer)
{
    ASSERT(layer != NULL, "Add empty layer into group");

    /* Dimension check */
    bool check = (layer->getIFMapSize() != NULL) && (layer->getOFMapSize() != NULL);
    ASSERT(check, "Add layer with empty I/O feature map");

    /* Handle the I/O feature map*/
    (groupType == Group_t::CaseCade) ? addCaseCade(move(layer)) : addCaseCode(move(layer));
}


/** ===============================================================================================
 * \name    addCaseCade
 *
 * \brief   Add layer into the end of the graph.
 * 
 * \endcond
 * ================================================================================================
 */
void::
LayerGroup::addCaseCade (Layer* layer)
{
    /* Handle the I/O feature map*/
    if (layers.size() == 0)
    {
        iFMapSize = layer->getIFMapSize();
        oFMapSize = layer->getOFMapSize();
        oFMap     = layer->getOFMap();

    } else {

        /* Dimension check */
        int* nextIFMapSize = layer->getIFMapSize();
        bool check = (oFMapSize[BATCH] == nextIFMapSize[BATCH]) && (oFMapSize[CHANNEL] == nextIFMapSize[CHANNEL]) && (oFMapSize[HEIGHT] == nextIFMapSize[HEIGHT]) && (oFMapSize[WIDTH] == nextIFMapSize[WIDTH]);
        ASSERT(check, "Layer " + to_string(layer->layerIndex) + " has error iFMapSize to the existing oFMapSize.");

        layer->setIFMap(oFMap);
        oFMapSize = layer->getOFMapSize();
        oFMap     = layer->getOFMap();
    }

    layers.emplace_back(layer);
}


/** ===============================================================================================
 * \name    addCaseCode
 *
 * \brief   Add layer next to the layers.
 * 
 * \endcond
 * ================================================================================================
 */
void::
LayerGroup::addCaseCode (Layer* layer)
{
    /* Handle the I/O feature map*/
    if (layers.size() == 0)
    {
        iFMapSize = layer->getIFMapSize();
        oFMapSize = layer->getOFMapSize();
        oFMap     = layer->getOFMap();

    } else {

        /* Dimension check */
        int* nextIFMapSize = layer->getIFMapSize();
        int* nextOFMapSize = layer->getOFMapSize();
        bool check  = (iFMapSize[BATCH] == nextIFMapSize[BATCH]) && (iFMapSize[CHANNEL] == nextIFMapSize[CHANNEL]) && (iFMapSize[HEIGHT] == nextIFMapSize[HEIGHT]) && (iFMapSize[WIDTH] == nextIFMapSize[WIDTH])
                   && (oFMapSize[BATCH] == nextOFMapSize[BATCH]) && (oFMapSize[CHANNEL] == nextOFMapSize[CHANNEL]) && (oFMapSize[HEIGHT] == nextOFMapSize[HEIGHT]) && (oFMapSize[WIDTH] == nextOFMapSize[WIDTH]);
        ASSERT(check, "Casecoded layer has error iFMapSize or oFMapSize to the existing layer");

        layer->setIFMap(iFMap);
    }

    layers.emplace_back(layer);
}


/** ===============================================================================================
 * \name    printInfo
 *
 * \brief   Print the group information.
 * 
 * \endcond
 * ================================================================================================
 */
void::
LayerGroup::printInfo ()
{
#if PRINT_MODEL_DETIAL
    cout << ((groupType == Group_t::CaseCade) ? "sequential" : "branch") << " start -------------" << std::endl;
    for (auto layer: layers){
        layer->printInfo();
    }
    cout << ((groupType == Group_t::CaseCade) ? "sequential" : "branch") << " end -------------" << std::endl;
#else
    std::cout << "(" 
              << std::right << std::setw(3)  << iFMapSize[BATCH]             << ", " \
              << std::right << std::setw(3)  << iFMapSize[CHANNEL]           << ", " \
              << std::right << std::setw(4)  << iFMapSize[HEIGHT]            << ", " \
              << std::right << std::setw(3)  << iFMapSize[WIDTH]                     \
              << std::left  << std::setw(10) << ")" << "("                           \
              << std::right << std::setw(3)  << oFMapSize[BATCH]             << ", " \
              << std::right << std::setw(3)  << oFMapSize[CHANNEL]           << ", " \
              << std::right << std::setw(4)  << oFMapSize[HEIGHT]            << ", " \
              << std::right << std::setw(3)  << oFMapSize[WIDTH]                     \
              << std::left  << std::setw(10) << ")"; 
              
    std::cout << std::endl;
#endif
}


/** ===============================================================================================
 * \name    issueLayer
 *
 * \brief   Print the group information.
 * 
 * \endcond
 * ================================================================================================
 */
void::
LayerGroup::issueLayer ()
{
    for (auto layer: layers){
        layer->issueLayer();
    }


}
