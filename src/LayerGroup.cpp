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
 * \param   group_type      Group_t::CaseCade | Group_t::CaseCode
 * 
 * \endcond
 * ================================================================================================
 */
LayerGroup::LayerGroup(Group_t group_type, char* layer_type): Layer(-1), groupType(group_type)
{

}


/** ===============================================================================================
 * \name   ~LayerGroup
 *
 * \brief   Destruct a layerGroup
 * 
 * \note    The I/O memory should release one by one with carefully consideration due to the pointer
 *          passing through the layer graph
 * 
 * \endcond
 * ================================================================================================
 */
LayerGroup::~LayerGroup()
{
    for (auto layer = layers.begin(); layer != layers.end(); ++layer) delete *layer;
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
    bool check = !layer->getIFMapSize().empty() && !layer->getOFMapSize().empty();
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
    if (layers.empty())
    {
        iFMapSize = layer->getIFMapSize();
        oFMapSize = layer->getOFMapSize();
        oFMap     = layer->getOFMap();

    } else {

        /* Dimension check */
        vector<int> nextIFMapSize = layer->getIFMapSize();
        bool check = (oFMapSize[BATCH] == nextIFMapSize[BATCH]) && (oFMapSize[CHANNEL] == nextIFMapSize[CHANNEL] && (oFMapSize[HEIGHT] == nextIFMapSize[HEIGHT]) && (oFMapSize[WIDTH] == nextIFMapSize[WIDTH]));
        ASSERT(check, "Layer " + to_string(layer->layerID) + " (" + layer->layerType + ") has error iFMapSize to the existing oFMapSize.");

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
    if (layers.empty())
    {
        iFMapSize = layer->getIFMapSize();
        oFMapSize = layer->getOFMapSize();
        oFMap     = layer->getOFMap();

    } else {

        /* Dimension check */
        vector<int> nextIFMapSize = layer->getIFMapSize();
        vector<int> nextOFMapSize = layer->getOFMapSize();
        bool check  = (iFMapSize[BATCH] == nextIFMapSize[BATCH]) && (iFMapSize[CHANNEL] == nextIFMapSize[CHANNEL]) && (iFMapSize[HEIGHT] == nextIFMapSize[HEIGHT]) && (iFMapSize[WIDTH] == nextIFMapSize[WIDTH])
                   && (oFMapSize[BATCH] == nextOFMapSize[BATCH]) && (oFMapSize[CHANNEL] == nextOFMapSize[CHANNEL]) && (oFMapSize[HEIGHT] == nextOFMapSize[HEIGHT]) && (oFMapSize[WIDTH] == nextOFMapSize[WIDTH]);
        ASSERT(check, "Casecoded layer has error iFMapSize or oFMapSize to the existing layer");

        layer->setIFMap(iFMap);
        layer->setOFMap(oFMap);
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
    if (LOG_LEVEL >= VERBOSE) std::cout << "oFMap ";
    if(oFMap.second)  mmu->memoryAllocate(oFMap.first,  oFMap.second->size()  * sizeof(DATA_TYPE));
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
LayerGroup::compileToKernel(int app_id, int model_id, vector<Kernel>& container, vector<Kernel*> dependency)
{
    log_V("LayerGroup", "compileToKernel");
    if (groupType == Group_t::CaseCade) {
        for (auto layer: layers)
        {
            dependency = layer->compileToKernel(app_id, model_id, container, dependency);
        }
    } else {  // groupType == Group_t::CaseCode
        vector<Kernel*> new_dependency;
        for (auto layer: layers)
        {
            vector<Kernel*> temp = layer->compileToKernel(app_id, model_id, container, dependency);
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
LayerGroup::setIFMap(pair<int, vector<DATA_TYPE>*> data)
{
    if (iFMap.second) delete iFMap.second;
    iFMap  = data;

    if (groupType == Group_t::CaseCade) 
    {
        layers.front()->setIFMap(data);

    } else {
        for (auto layer: layers) layer->setIFMap(data);
    }
}


/** ===============================================================================================
 * \name    setOFMap
 *
 * \brief   Set the output feature map
 * 
 * \endcond
 * ================================================================================================
 */
void
LayerGroup::setOFMap(pair<int, vector<DATA_TYPE>*> data)
{
    if (oFMap.second) delete oFMap.second;
    oFMap  = data;

    if (groupType == Group_t::CaseCade) 
    {
        layers.back()->setOFMap(data);

    } else {
        for (auto layer: layers) layer->setOFMap(data);
    }
}


/** ===============================================================================================
 * \name    setFilter
 *
 * \brief   Set the filter
 * 
 * \endcond
 * ================================================================================================
 */
void
LayerGroup::setFilter(pair<int, vector<DATA_TYPE>*> data)
{
    ASSERT(false, "Cannot set filter to a layerGroup");
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
    std::cout << ((groupType == Group_t::CaseCade) ? "sequential" : "branch") << " start -------------" << std::endl;
    for (auto layer: layers){
        layer->printInfo();
    }
    std::cout << ((groupType == Group_t::CaseCade) ? "sequential" : "branch") << " end -------------" << std::endl;
#else
    std::cout << "(" 
              << std::right << std::setw(3)  << iFMapSize[BATCH]   << ", "
              << std::right << std::setw(3)  << iFMapSize[CHANNEL] << ", "
              << std::right << std::setw(4)  << iFMapSize[HEIGHT]  << ", "
              << std::right << std::setw(3)  << iFMapSize[WIDTH]          
              << std::left  << std::setw(10) << ")" << "("                
              << std::right << std::setw(3)  << oFMapSize[BATCH]   << ", "
              << std::right << std::setw(3)  << oFMapSize[CHANNEL] << ", "
              << std::right << std::setw(4)  << oFMapSize[HEIGHT]  << ", "
              << std::right << std::setw(3)  << oFMapSize[WIDTH]          
              << std::left  << std::setw(10) << ")"; 
              
    std::cout << std::endl;
#endif
}



/** ===============================================================================================
 * \name    BasicBlock18
 *
 * \brief   The layergroup prototype used in ResNet18
 * 
 * \param   input_size          [batch, channel, height, width]
 * \param   down_sample         whether downsampling the dimension
 * 
 * \endcond
 * ================================================================================================
 */
ResNetBlock18::ResNetBlock18(int& layer_id, vector<int> input_size, bool down_sample) : LayerGroup(Group_t::CaseCode, (char*)"ResNetBlock18")
{
    down_sample ? BottleNeckBlock(layer_id, input_size) : BasicBlock(layer_id, input_size);
}


/** ===============================================================================================
 * \name    BasicBlock
 *
 * \brief   The basicblock for ResNet18
 * 
 * \param   input_size          [batch, channel, height, width]
 * 
 * \endcond
 * 
 * #Index    Type     Kernel    Feature     Output    Stride    Padding    Activation
 *                     Size       Map        Size
 *    |       Data                 c       n  x  n
 *   / \                                    
 *  |   1   Conv2D    3 x 3        c       n  x  n     1          1          ReLU
 *  |   2   Conv2D    3 x 3        c       n  x  n     1          1        
 *  3   |   ByPass                 c       n  x  n
 *   \ /
 *    |
 * ================================================================================================
 */
void
ResNetBlock18::BasicBlock(int& layer_id, vector<int> input_size)
{
    int channel = input_size[CHANNEL];
    LayerGroup* sequential = new LayerGroup();
    sequential->addLayer(new Conv2D(layer_id++, input_size, {channel, channel, 3, 3},  (char*)"ReLU", 1, 1));
    sequential->addLayer(new Conv2D(layer_id++, input_size, {channel, channel, 3, 3},  (char*)"None", 1, 1));
    
    addLayer(sequential);
    addLayer(new ByPass(layer_id++, input_size));
}


/** ===============================================================================================
 * \name    BasicBlock
 *
 * \brief   The basicblock for ResNet18
 * 
 * \param   input_size          [batch, channel, height, width]
 * 
 * \endcond
 * 
 * #Index    Type     Kernel    Feature     Output    Stride    Padding    Activation
 *                     Size       Map        Size
 *    |       Data                 c        n x n 
 *   / \                                    
 *  |   1   Conv2D    3 x 3        c      n/2 x n/2     2          1          ReLU
 *  |   2   Conv2D    3 x 3        c      n/2 x n/2     1          1        
 *  3   |   Conv2D    3 x 3        c      n/2 x n/2     2          1          ReLU
 *   \ /
 *    |
 * ================================================================================================
 */
void
ResNetBlock18::BottleNeckBlock(int& layer_id, vector<int> input_size)
{
    int channel = input_size[CHANNEL];
    LayerGroup* sequential = new LayerGroup();
    sequential->addLayer(new Conv2D(layer_id++, input_size, {channel * 2, channel, 3, 3},  (char*)"ReLU", 2, 1));
    sequential->addLayer(new Conv2D(layer_id++, sequential->getOFMapSize(), {channel * 2, channel * 2, 3, 3},  (char*)"None", 1, 1));

    addLayer(sequential);
    addLayer(new Conv2D(layer_id++, input_size, {channel * 2, channel, 3, 3},  (char*)"ReLU", 2, 1));
}



/** ===============================================================================================
 * \name    Inception
 *
 * \brief   The layergroup prototype used in GoogleNet
 * 
 * \param   input_size          [batch, channel, height, width]
 * \param   channel_1x1         the channel for the    1x1 layers
 * \param   channel_reduce_3x3  the channel for first  3x3 layers
 * \param   channel_3x3         the channel for second 3x3 layers
 * \param   channel_reduce_5x5  the channel for first  5x5 layers
 * \param   channel_5x5         the channel for second 5x5 layers
 * \param   channel_pooling     the channel for the pooling layers
 * 
 * \endcond
 * 
 *  #Index     Type   Kernel           Feature     Output    Stride    Padding    Activation
 *                     Size              Map        Size
 *     |       Data                       c         n x n
 *  / / \ \  
 * 1 |   | | Conv2D    1 x 1        channel_1x1     n x n      1          0          ReLU
 * | 3   | | Conv2D    1 x 1 channel_reduce_3x3     n x n      1          0          ReLU
 * | 4   | | Conv2D    3 x 3        channel_3x3     n x n      1          1          ReLU
 * | |   6 | Conv2D    1 x 1 channel_reduce_5x5     n x n      1          0          ReLU
 * | |   7 | Conv2D    5 x 5        channel_5x5     n x n      1          2          ReLU
 * | |   | 9   Pool    3 x 3              c         n x n      1          1           Max
 * | |   |10 Conv2D    1 x 1    channel_pooling     n x n      1          0          ReLU
 * | |   | |
 * 2 |   | | ByPass    channel_1x1 + channel_3x3 + channel_5x5 + channel_pooling     n x n
 * | 5   | | ByPass    channel_1x1 + channel_3x3 + channel_5x5 + channel_pooling     n x n
 * | |   8 | ByPass    channel_1x1 + channel_3x3 + channel_5x5 + channel_pooling     n x n
 * | |   |11 ByPass    channel_1x1 + channel_3x3 + channel_5x5 + channel_pooling     n x n
 *  \ \ / /
 *     |    
 * ================================================================================================
 */
Inception::Inception(int& layer_id, vector<int> input_size, int channel_1x1, int channel_reduce_3x3, int channel_3x3, int channel_reduce_5x5, int channel_5x5, int channel_pooling)
        : LayerGroup(Group_t::CaseCode, (char*)"Inception"), channel_1x1(channel_1x1), channel_reduce_3x3(channel_reduce_3x3), channel_3x3(channel_3x3)
        , channel_reduce_5x5(channel_reduce_5x5), channel_5x5(channel_5x5), channel_pooling(channel_pooling)
{
    int height = input_size[HEIGHT];
    int weight = input_size[WIDTH];
    int final_dim = channel_1x1 + channel_3x3 + channel_5x5 + channel_pooling;

    LayerGroup* sequential_1x1 = new LayerGroup();
    sequential_1x1->addLayer(new Conv2D(layer_id++, input_size, {channel_1x1, input_size[CHANNEL], 1, 1},  (char*)"ReLU", 1, 0));
    sequential_1x1->addLayer(new ByPass(layer_id++, sequential_1x1->getOFMapSize(), {input_size[BATCH], final_dim, height, weight}));

    LayerGroup* sequential_3x3 = new LayerGroup();
    sequential_3x3->addLayer(new Conv2D(layer_id++, input_size, {channel_reduce_3x3, input_size[CHANNEL], 1, 1} ,  (char*)"ReLU", 1, 0));
    sequential_3x3->addLayer(new Conv2D(layer_id++, sequential_3x3->getOFMapSize(), {channel_3x3, channel_reduce_3x3, 3, 3}        ,  (char*)"ReLU", 1, 1));
    sequential_3x3->addLayer(new ByPass(layer_id++, sequential_3x3->getOFMapSize(), {input_size[BATCH], final_dim, height, weight}));
    
    LayerGroup* sequential_5x5 = new LayerGroup();
    sequential_5x5->addLayer(new Conv2D(layer_id++, input_size, {channel_reduce_5x5, input_size[CHANNEL], 1, 1} ,  (char*)"ReLU", 1, 0));
    sequential_5x5->addLayer(new Conv2D(layer_id++, sequential_5x5->getOFMapSize(), {channel_5x5, channel_reduce_5x5, 5, 5}        ,  (char*)"ReLU", 1, 2));
    sequential_5x5->addLayer(new ByPass(layer_id++, sequential_5x5->getOFMapSize(), {input_size[BATCH], final_dim, height, weight}));
    
    LayerGroup* sequential_pooling = new LayerGroup();
    sequential_pooling->addLayer(new Pooling(layer_id++, input_size, {3, 3},  (char*)"Max", 1, 1));
    sequential_pooling->addLayer(new Conv2D(layer_id++, sequential_pooling->getOFMapSize(), {channel_pooling, input_size[CHANNEL], 1, 1},  (char*)"ReLU", 1, 0));
    sequential_pooling->addLayer(new ByPass(layer_id++, sequential_pooling->getOFMapSize(), {input_size[BATCH], final_dim, height, weight}));

    addLayer(sequential_1x1);
    addLayer(sequential_3x3);
    addLayer(sequential_5x5);
    addLayer(sequential_pooling);
}

