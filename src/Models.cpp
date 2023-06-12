/**
 * \name    Models.cpp
 * 
 * \brief   Implement the API
 * 
 * \date    APR 4, 2023
 */

#include "include/Models.hpp"

/* ************************************************************************************************
 * Global Variable
 * ************************************************************************************************
 */
int Model::modelCount = 0;


/** ===============================================================================================
 * \name    Model
 * 
 * \brief   Construct a model
 * 
 * \param   app_id          the application index
 * \param   model_type      the model type
 * \param   task            the task infomations include arrival_time, deadline, input_size, and data
 * 
 * \endcond
 * ================================================================================================
 */
Model::Model(int app_id, const char* model_type, Task task)
    : appID(app_id), modelType(model_type), modelID(modelCount++), task(task)
{
    modelGraph = new LayerGroup();
    startTime = total_gpu_cycle;

    buildLayerGraph();
}


/** ===============================================================================================
 * \name   ~Model
 * 
 * \brief   Destruct a model
 * 
 * \endcond
 * ================================================================================================
 */
Model::~Model()
{
    delete modelGraph;
    for (auto kernel : kernelContainer)
    {
        while(!kernel.requests.empty())
        {
            auto request = kernel.requests.front();
            kernel.requests.pop();
            delete request;
        }
    }
}


/** ===============================================================================================
 * \name    setBatchSize
 *
 * \brief   Change the batch size of model
 * 
 * \param   new_batch_size      the size of new batch
 * 
 * \endcond
 * ================================================================================================
 */
void
Model::setBatchSize (int batch_size) 
{
    task.inputSize[BATCH] = batch_size;
    modelGraph->changeBatch(batch_size);
}


/** ===============================================================================================
 * \name    memoryAllocate
 * 
 * \brief   Allocate physical address to the model virtual address. For the inference engine.
 * 
 * \param   mmu     the pointer of memory manager unit from CPU
 * 
 * \endcond
 * ================================================================================================
 */
void
Model::memoryAllocate(MMU* mmu)
{
    modelGraph->memoryAllocate(mmu);
}


/** ===============================================================================================
 * \name    memoryRelease
 * 
 * \brief   Release the used resource.
 * 
 * \param   mmu     the pointer of memory manager unit from CPU
 * 
 * \return  total page record of this model
 * 
 * \endcond
 * ================================================================================================
 */
PageRecord
Model::memoryRelease(MMU* mmu)
{
#if (RECORD_MODEL_INFORMATIONS)

    Model::ModelInfo info;
    unordered_set<int> io_va_list, filter_va_list;

    for(auto& kernel : kernelContainer) 
    {
        io_va_list.insert(kernel.srcLayer->getIFMap().first);
        io_va_list.insert(kernel.srcLayer->getOFMap().first);
        filter_va_list.insert(kernel.srcLayer->getFilter().first);

        info += kernel.getKernelInfo();
    }

    for (auto va : io_va_list) info.ioMemCount += mmu->lookup(va);
    for (auto va : filter_va_list) info.filterMemCount += mmu->lookup(va);

    std::cout << "Summary: " << modelType << " ("
              << std::right << std::setw(4)  << task.inputSize[BATCH]    << ", "
              << std::right << std::setw(4)  << task.inputSize[CHANNEL]  << ", "
              << std::right << std::setw(3)  << task.inputSize[HEIGHT]   << ", "
              << std::right << std::setw(3)  << task.inputSize[WIDTH] 
              << std::left  << std::setw(10) << ")" << std::endl;

    std::cout << std::left << std::setw(15) << "Num Layer"; 
    std::cout << std::left << std::setw(15) << "Request"; 
    std::cout << std::left << std::setw(15) << "ioMem"; 
    std::cout << std::left << std::setw(15) << "filterMem"; 
    std::cout << std::left << std::setw(15) << "Read"; 
    std::cout << std::left << std::setw(15) << "Write"; 
    std::cout << std::left << std::setw(15) << "Cycle"; 
    std::cout << std::endl;
    
    std::cout << std::left << std::setw(15) << kernelContainer.size(); 
    std::cout << std::left << std::setw(15) << info.numOfRequest; 
    std::cout << std::left << std::setw(15) << info.ioMemCount; 
    std::cout << std::left << std::setw(15) << info.filterMemCount; 
    std::cout << std::left << std::setw(15) << info.numOfRead; 
    std::cout << std::left << std::setw(15) << info.numOfWrite; 
    std::cout << std::left << std::setw(15) << info.numOfCycle; 
    std::cout << std::endl;
#endif

    for(auto& kernel : kernelContainer) 
    {
        page_record += kernel.memoryRelease(mmu);
    }

    return page_record;
}


/** ===============================================================================================
 * \name    compileToKernel
 * 
 * \brief   Make the kernel dependency.
 * 
 * \return  reference of model kernels
 * 
 * \endcond
 * ================================================================================================
 */
vector<Kernel>&
Model::compileToKernel()
{
    log_T("Model", "compileToKernel");
    
    kernelContainer.reserve(numOfLayer);
    auto dependencyKernels = modelGraph->compileToKernel(appID, modelID, kernelContainer, {});

#if (LOG_LEVEL >= VERBOSE)
    bool title = true;
    for (auto kernel : kernelContainer)
    {
        kernel.printInfo(title);
        title = false;
    }
#endif

    return kernelContainer;
}


/** ===============================================================================================
 * \name    findReadyKernels
 * 
 * \brief   Find all ready kernel of the model
 * 
 * \return  a list of ready kernel pointer
 * 
 * \endcond
 * ================================================================================================
 */
list<Kernel*>
Model::findReadyKernels()
{
    list<Kernel*> readyList;
    for (auto& kernel : kernelContainer)
    {
        if(!kernel.isFinish() && !kernel.isRunning() && kernel.isReady())
        {
            readyList.emplace_back(&kernel);
        }
        
    }
    return readyList;
}


/** ===============================================================================================
 * \name    getRunningKernels
 * 
 * \brief   Get all running kernels
 * 
 * \return  a list of running kernel pointer
 * 
 * \endcond
 * ================================================================================================
 */
list<Kernel*>
Model::getRunningKernels()
{
    list<Kernel*> runningList;
    for (auto& kernel : kernelContainer)
    {
        if(kernel.isRunning()) runningList.emplace_back(&kernel);
    }
    return runningList;
}


/** ===============================================================================================
 * \name    getKernelStatus
 * 
 * \brief   Get the finish flag of the kernel
 * 
 * \return  the list of finish flag
 * 
 * \endcond
 * ================================================================================================
 */
vector<bool>
Model::getKernelStatus()
{
    vector<bool> finishFlags;
    for (auto& kernel : kernelContainer) finishFlags.emplace_back(kernel.isFinish());

    return finishFlags;
}


/** ===============================================================================================
 * \name    checkFinish
 * 
 * \brief   Construct a model
 * 
 * \endcond
 * ================================================================================================
 */
bool
Model::checkFinish()
{
    return kernelContainer.back().isFinish();
}


/** ===============================================================================================
 * \name    getModelInfo
 * 
 * \brief   Return the pre-collected model information
 * 
 * \return  Model::ModelInfo
 * 
 * \note    Can open the \b RECORD_MODEL_INFORMATIONS flag in App_config.h to extract the infomation
 * 
 * \endcond
 * ================================================================================================
 */
Model::ModelInfo
Model::getModelInfo(const char* model_type)
{
    ModelInfo Info (model_type);

    if (strcmp(model_type, "LeNet") == 0) {
        Info.numOfLayers    = 8;
        Info.numOfRequest   = 8494;
        Info.ioMemCount     = 9518;
        Info.filterMemCount = 62638;
        Info.numOfRead      = 941088;
        Info.numOfWrite     = 8494;
        Info.numOfCycle     = 170688;
        Info.inputSize      = {1, 32, 32};
        Info.outputSize     = {1000};
        Info.totalExecuteTime = 479379;
        Info.layerExecuteTime = {
            56530, 55802, 55840, 55806, 83334, 59727, 56516, 55814
        };

    } else if (strcmp(model_type, "CaffeNet") == 0) {
        Info.numOfLayers    = 12;
        Info.numOfRequest   = 158824;
        Info.ioMemCount     = 171368;
        Info.filterMemCount = 30075936;
        Info.numOfRead      = 451046656;
        Info.numOfWrite     = 158824;
        Info.numOfCycle     = 3094016;
        Info.inputSize      = {3, 112, 112};
        Info.outputSize     = {1000};
        Info.totalExecuteTime = 6083439;
        Info.layerExecuteTime = {
            293484, 90747, 191013, 133107, 215017, 308223, 214664, 161354, 27902, 775009, 2933716, 739188
        };

    } else if (strcmp(model_type, "ResNet18") == 0) {
        Info.numOfLayers    = 28;
        Info.numOfRequest   = 828904;
        Info.ioMemCount     = 649448;
        Info.filterMemCount = 21992640;
        Info.numOfRead      = 1050410496;
        Info.numOfWrite     = 828904;
        Info.numOfCycle     = 8384256;
        Info.inputSize      = {3, 112, 112};
        Info.outputSize     = {1000};
        Info.totalExecuteTime = 6083439;
        Info.layerExecuteTime = {
            157843, 90189, 90457, 90456, 25, 90457, 90456, 25, 91516, 97921, 34067, 97922, 97921, 13, 109571, 
            162413, 53116, 162413, 162413, 7, 240080, 452461, 212131, 452462, 452461, 7, 1494450, 226377
        };

    } else if (strcmp(model_type, "VGG16") == 0) {
        Info.numOfLayers    = 22;
        Info.numOfRequest   = 3781608;
        Info.ioMemCount     = 3794152;
        Info.filterMemCount = 56899264;
        Info.numOfRead      = 7931585792;
        Info.numOfWrite     = 3781608;
        Info.numOfCycle     = 35397120;
        Info.inputSize      = {3, 112, 112};
        Info.outputSize     = {1000};
        Info.totalExecuteTime = 15634243;
        Info.layerExecuteTime = {
            875676, 209785, 74301, 151180, 157586, 70055, 126374, 205839, 205839, 82806, 303281, 488122, 488122, 
            258090, 454227, 454227, 454227, 256325, 84040, 2933716, 2933716, 739188
        };

    } else if (strcmp(model_type, "GoogleNet") == 0) {
        Info.numOfLayers    = 108;
        Info.numOfRequest   = 1198778;
        Info.ioMemCount     = 1254762;
        Info.filterMemCount = 44561920;
        Info.numOfRead      = 2045716128;
        Info.numOfWrite     = 1198778;
        Info.numOfCycle     = 9824148;
        Info.inputSize      = {3, 112, 112};
        Info.outputSize     = {1000};
        Info.totalExecuteTime = 5209661;
        Info.layerExecuteTime = {
            157843, 90189, 60079, 105368, 169833, 57585, 28961, 99033, 94719, 56149, 117258, 84454, 7, 88443, 56153, 
            7, 172740, 30026, 110734, 105335, 56149, 128189, 89783, 7, 96967, 56863, 7, 1122077, 64328, 28608, 147429, 
            99308, 7, 183210, 84807, 7, 134531, 55804, 7, 174500, 28255, 155907, 131027, 27902, 247194, 86590, 7, 135971, 
            55804, 7, 173088, 28255, 162261, 109571, 28255, 247194, 86590, 7, 135971, 55804, 7, 144487, 28255, 156966, 
            137379, 28255, 256019, 87665, 7, 135971, 55804, 7, 151902, 28961, 204674, 151470, 7, 258307, 91901, 7, 164070, 
            57224, 7, 1394219, 73903, 27902, 269896, 150764, 27902, 388402, 64350, 7, 194066, 56871, 7, 249279, 28255, 
            262908, 195602, 7, 416991, 68616, 7, 194066, 56871, 7, 3044330, 226377
        };

    }

    return Info;
}


/** ===============================================================================================
 * \name    buildLayerGraph
 * 
 * \brief   Build the model graph
 * 
 * \param   model_type   the choosen model graph name
 * 
 * \endcond
 * ================================================================================================
 */
void
Model::buildLayerGraph()
{
    log_T("Model", "buildLayerGraph");

    if (strcmp(modelType, "LeNet") == 0) {
        LeNet(task.inputSize);

    } else if (strcmp(modelType, "CaffeNet") == 0) {
        CaffeNet(task.inputSize);

    } else if (strcmp(modelType, "ResNet18") == 0) {
        ResNet18(task.inputSize);

    } else if (strcmp(modelType, "VGG16") == 0) {
        VGG16(task.inputSize);

    } else if (strcmp(modelType, "GoogleNet") == 0) {
        GoogleNet(task.inputSize);

    }
}


/** ===============================================================================================
 * \name    LeNet
 * 
 * \brief   Build the test graph
 * 
 * \param   imput_size  [batch, channel, height, width]
 * 
 * \endcond
 * 
 * #Index    Type    Kernel    Feature      Output   Stride    Padding    Activation
 *                    Size       Map         Size
 *                                1        32 x 32 
 *    1     Conv2D    5 x 5       6        28 x 28     1          0          Tanh
 *    2       Pool    2 x 2       6        14 x 14     2          0
 *    3     Conv2D    5 x 5      16        10 x 10     1          0          Tanh
 *    4       Pool    2 x 2      16         5 x 5      2          0
 *    5    Flatten              400         1 x 1
 *    6      Dense    1 x 1     120         1 x 1                            Tanh
 *    7      Dense    1 x 1      84         1 x 1                            Tanh
 *    8      Dense    1 x 1      10         1 x 1                         SoftMax
 * ================================================================================================
 */
void 
Model::LeNet(vector<int> input_size)
{
    int layer_id = 0;

    modelGraph->addLayer(new Conv2D (layer_id++, input_size, {6, 1, 5, 5}, (char*)"Tanh", 1, 0));
    modelGraph->addLayer(new Pooling(layer_id++, modelGraph->getOFMapSize(), { 6,  6, 2, 2}, (char*)"None", 2, 0));
    modelGraph->addLayer(new Conv2D (layer_id++, modelGraph->getOFMapSize(), {16,  6, 5, 5}, (char*)"Tanh", 1, 0));
    modelGraph->addLayer(new Pooling(layer_id++, modelGraph->getOFMapSize(), {16, 16, 2, 2}, (char*)"None", 2, 0));
                                                      
    modelGraph->addLayer(new Flatten(layer_id++, modelGraph->getOFMapSize()));
                                                        
    modelGraph->addLayer(new Dense(layer_id++, modelGraph->getOFMapSize(), 120));
    modelGraph->addLayer(new Dense(layer_id++, modelGraph->getOFMapSize(),  84));
    modelGraph->addLayer(new Dense(layer_id++, modelGraph->getOFMapSize(),  10));

    numOfLayer = 8;
    
#if (PRINT_MODEL_DETIAL)
    printSummary();
#endif

    compileToKernel();

}


/** ===============================================================================================
 * \name    CaffeNet
 * 
 * \brief   Build the CaffeNet graph
 * 
 * \param   imput_size  [batch, channel, height, width]
 * 
 * \endcond
 * 
 * #Index    Type    Kernel    Feature      Output   Stride    Padding    Activation
 *                    Size       Map         Size
 *                                3       227 x 227
 *    1     Conv2D   11 x 11     96        55 x 55     4          0          ReLU
 *    2       Pool    3 x 3      96        27 x 27     2          0
 *    3     Conv2D    5 x 5     256        27 x 27     1          2          ReLU
 *    4       Pool    3 x 3     256        13 x 13     2          0
 *    5     Conv2D    3 x 3     384        13 x 13     1          1          ReLU
 *    6     Conv2D    3 x 3     384        13 x 13     1          1          ReLU
 *    7     Conv2D    3 x 3     256        13 x 13     1          1          ReLU
 *    8       Pool    3 x 3     256         6 x 6      2          0
 *    9    Flatten             9216         1 x 1
 *   10      Dense    1 x 1    4096         1 x 1                            ReLU
 *   11      Dense    1 x 1    4096         1 x 1                            ReLU 
 *   12      Dense    1 x 1    1000         1 x 1                         softmax
 * ================================================================================================
 */
void 
Model::CaffeNet(vector<int> input_size)
{
    int layer_id = 0;
    
    modelGraph->addLayer(new Conv2D (layer_id++, input_size, {96, 3, 11, 11}, (char*)"ReLU", 4, 0));
    modelGraph->addLayer(new Pooling(layer_id++, modelGraph->getOFMapSize(), { 96,  96,  3,  3}, (char*)"None", 2, 0));
    modelGraph->addLayer(new Conv2D (layer_id++, modelGraph->getOFMapSize(), {256,  96,  5,  5}, (char*)"ReLU", 1, 2));
    modelGraph->addLayer(new Pooling(layer_id++, modelGraph->getOFMapSize(), {256, 256,  3,  3}, (char*)"None", 2, 0));
    modelGraph->addLayer(new Conv2D (layer_id++, modelGraph->getOFMapSize(), {384, 256,  3,  3}, (char*)"ReLU", 1, 1));
    modelGraph->addLayer(new Conv2D (layer_id++, modelGraph->getOFMapSize(), {384, 384,  3,  3}, (char*)"ReLU", 1, 1));
    modelGraph->addLayer(new Conv2D (layer_id++, modelGraph->getOFMapSize(), {256, 384,  3,  3}, (char*)"ReLU", 1, 1));
    modelGraph->addLayer(new Pooling(layer_id++, modelGraph->getOFMapSize(), {256, 256,  3,  3}, (char*)"None", 2, 0));
                                                      
    modelGraph->addLayer(new Flatten(layer_id++, modelGraph->getOFMapSize()));
                                                        
    modelGraph->addLayer(new Dense(layer_id++, modelGraph->getOFMapSize(), 4096));
    modelGraph->addLayer(new Dense(layer_id++, modelGraph->getOFMapSize(), 4096));
    modelGraph->addLayer(new Dense(layer_id++, modelGraph->getOFMapSize(), 1000));

    numOfLayer = 12;
    
#if (PRINT_MODEL_DETIAL)
    printSummary();
#endif

    compileToKernel();

}


/** ===============================================================================================
 * \name    ResNet18
 * 
 * \brief   Build the ResNet18 layer graph
 * 
 * \param   imput_size  [batch, channel, height, width]
 * 
 * \endcond
 * 
 * #Index          Type   Kernel    Feature     Output    Stride    Padding    Activation
 *                         Size       Map        Size
 *                 Data                3       224 x 224
 *    1          Conv2D    7 x 7      64       112 x 112    2          3          ReLU
 *    2            Pool    3 x 3      64        56 x 56     2          1           Max  
 *    3      BasicBlock               64        56 x 56
 *    6      BasicBlock               64        56 x 56   
 *    9 BottleNeckBlock              128        28 x 28
 *   12      BasicBlock              128        28 x 28
 *   15 BottleNeckBlock              256        14 x 14
 *   18      BasicBlock              256        14 x 14
 *   21 BottleNeckBlock              512         7 x 7
 *   24      BasicBlock              512         7 x 7
 *   27            Pool    7 x 7    1024         1 x 1      2          1           Avg
 *   28           Dense    1 x 1    1000         1 x 1
 * ================================================================================================
 */
void
Model::ResNet18(vector<int> input_size)
{
    int layer_id = 0;

    modelGraph->addLayer(new Conv2D (layer_id++, input_size, {64, 3, 7, 7}, (char*)"ReLU", 2, 3));
    modelGraph->addLayer(new Pooling(layer_id++, modelGraph->getOFMapSize(), {64, 64, 3, 3}, (char*)"Max_Pool", 2, 1));

    modelGraph->addLayer(new ResNetBlock18(layer_id, modelGraph->getOFMapSize(), false));
    modelGraph->addLayer(new ResNetBlock18(layer_id, modelGraph->getOFMapSize(), false));

    modelGraph->addLayer(new ResNetBlock18(layer_id, modelGraph->getOFMapSize(), true));
    modelGraph->addLayer(new ResNetBlock18(layer_id, modelGraph->getOFMapSize(), false));

    modelGraph->addLayer(new ResNetBlock18(layer_id, modelGraph->getOFMapSize(), true));
    modelGraph->addLayer(new ResNetBlock18(layer_id, modelGraph->getOFMapSize(), false));

    modelGraph->addLayer(new ResNetBlock18(layer_id, modelGraph->getOFMapSize(), true));
    modelGraph->addLayer(new ResNetBlock18(layer_id, modelGraph->getOFMapSize(), false));

    int filter_size = modelGraph->getOFMapSize()[WIDTH];
    modelGraph->addLayer(new Pooling(layer_id++, modelGraph->getOFMapSize(), {1024, 512, filter_size, filter_size}, (char*)"Avg_Pool", 2, 0));
    modelGraph->addLayer(new Dense(layer_id++, modelGraph->getOFMapSize(), 1000));

    numOfLayer = 28;

#if (PRINT_MODEL_DETIAL)
    printSummary();
#endif
    
    compileToKernel();
}


/** ===============================================================================================
 * \name    VGG16
 * 
 * \brief   Build the VGG16 layer graph
 * 
 * \param   imput_size  [batch, channel, height, width]
 * 
 * \endcond
 * 
 * #Index    Type    Kernel    Feature     Output    Stride    Padding    Activation
 *                    Size       Map        Size
 *            Data                3       224 x 224
 *    1     Conv2D    3 x 3      64       224 x 224    1          1          ReLU
 *    2     Conv2D    3 x 3      64       224 x 224    1          1          ReLU
 *    3       Pool    2 x 2      64       112 x 112    2          0
 *    4     Conv2D    3 x 3     128       112 x 112    1          1          ReLU
 *    5     Conv2D    3 x 3     128       112 x 112    1          1          ReLU
 *    6       Pool    2 x 2     128        56 x 56     2          0
 *    7     Conv2D    3 x 3     256        56 x 56     1          1          ReLU
 *    8     Conv2D    3 x 3     256        56 x 56     1          1          ReLU
 *    9     Conv2D    3 x 3     256        56 x 56     1          1          ReLU
 *   10       Pool    2 x 2     256        28 x 28     2          0
 *   11     Conv2D    3 x 3     512        28 x 28     1          1          ReLU
 *   12     Conv2D    3 x 3     512        28 x 28     1          1          ReLU
 *   13     Conv2D    3 x 3     512        28 x 28     1          1          ReLU
 *   14       Pool    2 x 2     512        14 x 14     2          0
 *   15     Conv2D    3 x 3     512        14 x 14     1          1          ReLU
 *   16     Conv2D    3 x 3     512        14 x 14     1          1          ReLU
 *   17     Conv2D    3 x 3     512        14 x 14     1          1          ReLU
 *   18       Pool    2 x 2     512         7 x 7      2          0
 *   19    Flatten            25088         1 x 1
 *   20      Dense    1 x 1    4096         1 x 1                            ReLU
 *   21      Dense    1 x 1    4096         1 x 1                            ReLU 
 *   22      Dense    1 x 1    1000         1 x 1    
 * 
 * ================================================================================================
 */
void
Model::VGG16(vector<int> input_size)
{
    int layer_id = 0;
    
    modelGraph->addLayer(new Conv2D (layer_id++, input_size, {64, 3, 3, 3}, (char*)"ReLU", 1, 1));
    modelGraph->addLayer(new Conv2D (layer_id++, modelGraph->getOFMapSize(), {64, 64, 3, 3}, (char*)"ReLU", 1, 1));
    modelGraph->addLayer(new Pooling(layer_id++, modelGraph->getOFMapSize(), {64, 64, 2, 2}, (char*)"None", 2, 0));
                                                      
    modelGraph->addLayer(new Conv2D (layer_id++, modelGraph->getOFMapSize(), {128,  64, 3, 3}, (char*)"ReLU", 1, 1));
    modelGraph->addLayer(new Conv2D (layer_id++, modelGraph->getOFMapSize(), {128, 128, 3, 3}, (char*)"ReLU", 1, 1));
    modelGraph->addLayer(new Pooling(layer_id++, modelGraph->getOFMapSize(), {128, 128, 2, 2}, (char*)"None", 2, 0));
                                                      
    modelGraph->addLayer(new Conv2D (layer_id++, modelGraph->getOFMapSize(), {256, 128, 3, 3}, (char*)"ReLU", 1, 1));
    modelGraph->addLayer(new Conv2D (layer_id++, modelGraph->getOFMapSize(), {256, 256, 3, 3}, (char*)"ReLU", 1, 1));
    modelGraph->addLayer(new Conv2D (layer_id++, modelGraph->getOFMapSize(), {256, 256, 3, 3}, (char*)"ReLU", 1, 1));
    modelGraph->addLayer(new Pooling(layer_id++, modelGraph->getOFMapSize(), {256, 256, 2, 2}, (char*)"None", 2, 0));
                                                      
    modelGraph->addLayer(new Conv2D (layer_id++, modelGraph->getOFMapSize(), {512, 256, 3, 3}, (char*)"ReLU", 1, 1));
    modelGraph->addLayer(new Conv2D (layer_id++, modelGraph->getOFMapSize(), {512, 512, 3, 3}, (char*)"ReLU", 1, 1));
    modelGraph->addLayer(new Conv2D (layer_id++, modelGraph->getOFMapSize(), {512, 512, 3, 3}, (char*)"ReLU", 1, 1));
    modelGraph->addLayer(new Pooling(layer_id++, modelGraph->getOFMapSize(), {512, 512, 2, 2}, (char*)"None", 2, 0));
                                                      
    modelGraph->addLayer(new Conv2D (layer_id++, modelGraph->getOFMapSize(), {512, 512, 3, 3}, (char*)"ReLU", 1, 1));
    modelGraph->addLayer(new Conv2D (layer_id++, modelGraph->getOFMapSize(), {512, 512, 3, 3}, (char*)"ReLU", 1, 1));
    modelGraph->addLayer(new Conv2D (layer_id++, modelGraph->getOFMapSize(), {512, 512, 3, 3}, (char*)"ReLU", 1, 1));
    modelGraph->addLayer(new Pooling(layer_id++, modelGraph->getOFMapSize(), {512, 512, 2, 2}, (char*)"None", 2, 0));
                                                      
    modelGraph->addLayer(new Flatten(layer_id++, modelGraph->getOFMapSize()));
                                                        
    modelGraph->addLayer(new Dense(layer_id++, modelGraph->getOFMapSize(), 4096));
    modelGraph->addLayer(new Dense(layer_id++, modelGraph->getOFMapSize(), 4096));
    modelGraph->addLayer(new Dense(layer_id++, modelGraph->getOFMapSize(), 1000));

    numOfLayer = 22;
    
#if (PRINT_MODEL_DETIAL)
    printSummary();
#endif

    compileToKernel();

}


/** ===============================================================================================
 * \name    GoogleNet
 * 
 * \brief   Build the GoogleNet layer graph
 * 
 * \param   imput_size  [batch, channel, height, width]
 * 
 * \endcond
 * 
 * #Index     Type    Kernel    Feature     Output    Stride    Padding    Activation
 *                    Size       Map         Size
 *            Data                3       224 x 224
 *    1     Conv2D    7 x 7      64       112 x 112    2          3          ReLU
 *    2       Pool    3 x 3      64        56 x 56     2          1           Max
 *    3     Conv2D    1 x 1      64        56 x 56     1          1          ReLU
 *    4     Conv2D    3 x 3     192        56 x 56     1          0          ReLU
 *    5       Pool    3 x 3     192        28 x 28     2          1           Max
 *    6  Inception              256        28 x 28                           ReLU
 *                              {64,  96, 128,  16,  32,  32} 
 *   17  Inception              480        28 x 28                           ReLU
 *                              {128, 128, 192,  32,  96,  64} 
 *
 *   28       Pool    3 x 3     480        14 x 14     2          1           Max
 *   29  Inception              512        14 x 14                           ReLU
 *                              {192,  96, 208,  16,  48,  64} 
 *
 *   40  Inception              512        14 x 14                           ReLU
 *                              {160, 112, 224,  24,  64,  64} 
 *
 *   51  Inception              512        14 x 14                           ReLU
 *                              {128, 128, 256,  24,  64,  64} 
 *
 *   62  Inception              528        14 x 14                           ReLU
 *                              {112, 144, 288,  32,  64,  64} 
 *
 *   73  Inception              832        14 x 14                           ReLU
 *                              {256, 160, 320,  32, 128, 128} 
 *
 *   84       Pool    3 x 3     832         7 x 7      2          1           Max
 *   85  Inception              832         7 x 7                            ReLU
 *                              {256, 160, 320,  32, 128, 128} 
 *
 *   96  Inception             1024         7 x 7                            ReLU
 *                              {384, 192, 384,  48, 128, 128} 
 *  107       Pool    3 x 3     832         7 x 7      2          1           AVG
 *  108      Dense    1 x 1    1000         1 x 1                         SoftMax
 * ================================================================================================
 */
void 
Model::GoogleNet(vector<int> input_size)
{
    int layer_id = 0;
    
    modelGraph->addLayer(new Conv2D (layer_id++,input_size, {64, 3, 7, 7}, (char*)"ReLU",     2, 3));
    modelGraph->addLayer(new Pooling(layer_id++, modelGraph->getOFMapSize(), { 64,  64, 3, 3}, (char*)"Max_Pool", 2, 1));
    modelGraph->addLayer(new Conv2D (layer_id++, modelGraph->getOFMapSize(), { 64,  64, 1, 1}, (char*)"ReLU",     1, 0));
    modelGraph->addLayer(new Conv2D (layer_id++, modelGraph->getOFMapSize(), {192,  64, 3, 3}, (char*)"ReLU",     1, 1));
    modelGraph->addLayer(new Pooling(layer_id++, modelGraph->getOFMapSize(), {192, 192, 3, 3}, (char*)"Max_Pool", 2, 1));
    
    modelGraph->addLayer(new Inception(layer_id, modelGraph->getOFMapSize(), 64, 96, 128, 16, 32, 32));
    modelGraph->addLayer(new Inception(layer_id, modelGraph->getOFMapSize(), 128, 128, 192, 32, 96, 64));
    modelGraph->addLayer(new Pooling  (layer_id++, modelGraph->getOFMapSize(), {480, 480, 3, 3}, (char*)"Max_Pool", 2, 1));

    modelGraph->addLayer(new Inception(layer_id, modelGraph->getOFMapSize(), 192, 96, 208, 16, 48, 64));
    modelGraph->addLayer(new Inception(layer_id, modelGraph->getOFMapSize(), 160, 112, 224, 24, 64, 64));
    modelGraph->addLayer(new Inception(layer_id, modelGraph->getOFMapSize(), 128, 128, 256, 24, 64, 64));
    modelGraph->addLayer(new Inception(layer_id, modelGraph->getOFMapSize(), 112, 114, 288, 32, 64, 64));
    modelGraph->addLayer(new Inception(layer_id, modelGraph->getOFMapSize(), 256, 160, 320, 32, 128, 128));
    modelGraph->addLayer(new Pooling  (layer_id++, modelGraph->getOFMapSize(), {832, 832, 3, 3}, (char*)"Max_Pool", 2, 1));

    modelGraph->addLayer(new Inception(layer_id, modelGraph->getOFMapSize(), 256, 160, 320, 32, 128, 128));
    modelGraph->addLayer(new Inception(layer_id, modelGraph->getOFMapSize(), 384, 192, 384, 48, 128, 128));

    int filter_size = modelGraph->getOFMapSize()[WIDTH];
    modelGraph->addLayer(new Pooling  (layer_id++, modelGraph->getOFMapSize(), {1024, 1024, filter_size, filter_size}, (char*)"Avg_Pool", 2, 0));
   
    modelGraph->addLayer(new Dense(layer_id++, modelGraph->getOFMapSize(), 1000));

    numOfLayer = 108;
    
#if (PRINT_MODEL_DETIAL)
    printSummary();
#endif

    compileToKernel();

}



/** ===============================================================================================
 * \name    printSummary
 * 
 * \brief   Construct a model
 * 
 * \endcond
 * ================================================================================================
 */
void
Model::printSummary()
{
    std::cout << "Model " << modelID << ", " << modelType << " summary:" << std::endl;
    std::cout << std::left << std::setw(10)  << "Layer_ID"    \
              << std::left << std::setw(12)  << "Layer_Type"  \
              << std::left << std::setw(25)  << "Activation_Type" \
              << std::left << std::setw(30)  << "Input_Size" \
              << std::left << std::setw(30)  << "Filter_Size" \
              << std::left << std::setw(26)  << "Output_Size" \
              << std::left << std::setw(17)  << "Stride" \
              << std::left << std::setw(10)  << "Padding" ;

    std::cout << std::endl;

    modelGraph->printInfo();
    
}