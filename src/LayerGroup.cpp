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
 * \param   group_type      Group_t::CaseCade / Group_t::CaseCode
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
        vector<int>* nextIFMapSize = layer->getIFMapSize();
        bool check = ((*oFMapSize)[BATCH] == (*nextIFMapSize)[BATCH]) && ((*oFMapSize)[CHANNEL] == (*nextIFMapSize)[CHANNEL]) && ((*oFMapSize)[HEIGHT] == (*nextIFMapSize)[HEIGHT]) && ((*oFMapSize)[WIDTH] == (*nextIFMapSize)[WIDTH]);
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

        int size = (*oFMapSize)[BATCH] * (*oFMapSize)[CHANNEL] * (*oFMapSize)[HEIGHT] * (*oFMapSize)[WIDTH];
        oFMap = new vector<unsigned char>(size);

    } else {

        /* Dimension check */
        vector<int>* nextIFMapSize = layer->getIFMapSize();
        vector<int>* nextOFMapSize = layer->getOFMapSize();
        bool check  = ((*iFMapSize)[BATCH] == (*nextIFMapSize)[BATCH]) && ((*iFMapSize)[CHANNEL] == (*nextIFMapSize)[CHANNEL]) && ((*iFMapSize)[HEIGHT] == (*nextIFMapSize)[HEIGHT]) && ((*iFMapSize)[WIDTH] == (*nextIFMapSize)[WIDTH])
                   && ((*oFMapSize)[BATCH] == (*nextOFMapSize)[BATCH]) && ((*oFMapSize)[CHANNEL] == (*nextOFMapSize)[CHANNEL]) && ((*oFMapSize)[HEIGHT] == (*nextOFMapSize)[HEIGHT]) && ((*oFMapSize)[WIDTH] == (*nextOFMapSize)[WIDTH]);
        ASSERT(check, "Casecoded layer has error iFMapSize or oFMapSize to the existing layer");

        layer->setIFMap(iFMap);
    }
    
    layers.emplace_back(layer);
}


/** ===============================================================================================
 * \name    changeBatch
 *
 * \brief   Change the batch size of model
 * 
 * \param   new_batch_size      the size of new batch
 * 
 * \endcond
 * ================================================================================================
 */
void
LayerGroup::changeBatch(int new_batch_size)
{
    for (auto layer: layers){
        layer->changeBatch(new_batch_size);
    }
}


/** ===============================================================================================
 * \name    memoryAllocate
 *
 * \brief   Allocate physical address to the model virtual address.
 * 
 * \param   mmu     the memory manager unit
 * 
 * \endcond
 * ================================================================================================
 */
void
LayerGroup::memoryAllocate(MMU* mmu)
{
    for (auto layer: layers){
        layer->memoryAllocate(mmu);
    }

    /* The memory sapce for merge layer */
    if (LOG_LEVEL >= VERBOSE) cout << "oFMap ";
    if(oFMap)  mmu->memoryAllocate(static_cast<int>(reinterpret_cast<std::uintptr_t>(oFMap)),  oFMap->size()  * sizeof(unsigned char));
}


/** ===============================================================================================
 * \name    compileToKernel
 *
 * \brief   Make the kernel dependency.
 * 
 * \param   container   the container to keep the compiled GPU requests
 * \param   dependency  the depended pointers of this kernel
 * 
 * \return  dependency of the next layer
 * 
 * \endcond
 * ================================================================================================
 */
vector<Kernel*> 
LayerGroup::compileToKernel(vector<Kernel>& container, vector<Kernel*> dependency)
{
    log_D("LayerGroup", "issueLayer");
    if (groupType == Group_t::CaseCade) {
        for (auto layer: layers)
        {
            dependency = layer->compileToKernel(container, dependency);
        }
    } else {  // groupType == Group_t::CaseCode
        vector<Kernel*> new_dependency;
        for (auto layer: layers)
        {
            vector<Kernel*> temp = layer->compileToKernel(container, dependency);
            new_dependency.insert(new_dependency.end(), temp.begin(), temp.end());
        }
        dependency = move(new_dependency);
    }

    return move(dependency);
}


/** ===============================================================================================
 * \name    setIFMap
 *
 * \brief   Set the input feature map
 * 
 * \endcond
 * ================================================================================================
 */
void
LayerGroup::setIFMap(vector<unsigned char>* data)
{
    if (iFMap != nullptr) delete iFMap;
    iFMap  = data;

    if (groupType == Group_t::CaseCade) 
    {
        auto layer = layers.front();
        layer->setIFMap(data);

    } else {
        for (auto layer: layers) {
            layer->setIFMap(data);
        }
    }
}


/** ===============================================================================================
 * \name    setFilter
 *
 * \brief   Set the output feature map
 * 
 * \endcond
 * ================================================================================================
 */
void
LayerGroup::setFilter(vector<unsigned char>* data)
{
    if (filter != nullptr) delete filter;
    filter = data;

    if (groupType == Group_t::CaseCade) 
    {
        auto layer = layers.front();
        layer->setFilter(data);

    } else {
        for (auto layer: layers) {
            layer->setFilter(data);
        }
    }
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
 * \name    init
 *
 * \brief   Initial the oFMapSize, defualt height and width is "1"
 * 
 * \endcond
 * 
 * ================================================================================================
 */
void
LayerGroup::calculateOFMapSize()
{
    
}
