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
 * \param   input_size      [batch = 1, channel, height, width]
 * \param   batch_size      the number of launched tasks when arrival
 * 
 * \endcond
 * ================================================================================================
 */
Model::Model(int app_id, const char* model_type, vector<int> input_size, int batch_size)
    : appID(app_id), modelType(model_type), modelID(modelCount++), inputSize(input_size), batchSize(batch_size)
{
    ASSERT(input_size[BATCH] == 1);
    modelGraph = new LayerGroup();
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
    batchSize = batch_size;
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
 * \endcond
 * ================================================================================================
 */
void
Model::memoryRelease(MMU* mmu)
{
    PageRecord record;
    for(auto& kernel : kernelContainer)
    {
        record += kernel.memoryRelease(mmu);
    }
    ofstream file(LOG_OUT_PATH + program_name + ".txt", std::ios::app);
        file << "PageRecord: [" << record.read_counter << ", " << record.write_counter << ", " << record.access_count << ", " << record.swap_count << "]" << endl;
    file.close();
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
 * \brief   Make the kernel dependency.
 * 
 * \return  reference of model kernels
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
 * \endcond
 * ================================================================================================
 */
Model::ModelInfo
Model::getModelInfo(const char* model_type)
{
    ModelInfo Info (model_type);

    if (strcmp(model_type, "LeNet") == 0) {
        Info.numOfLayers    = 8;
        Info.ioMemCount     = 9518;
        Info.filterMemCount = 62638;
        Info.inputSize      = {1, 32, 32};
        Info.outputSize     = {1000};

    } else if (strcmp(model_type, "CaffeNet") == 0) {
        Info.numOfLayers    = 12;
        Info.ioMemCount     = -1;
        Info.filterMemCount = -1;
        Info.inputSize      = {3, 227, 227};
        Info.outputSize     = {1000};

    } else if (strcmp(model_type, "ResNet18") == 0) {
        Info.numOfLayers    = 28;
        Info.ioMemCount     = 4166632;
        Info.filterMemCount = 39294144;
        Info.inputSize      = {3, 224, 224};
        Info.outputSize     = {1000};

    } else if (strcmp(model_type, "VGG16") == 0) {
        Info.numOfLayers    = 22;
        Info.ioMemCount     = 15262696;
        Info.filterMemCount = 140785344;
        Info.inputSize      = {3, 224, 224};
        Info.outputSize     = {1000};

    } else if (strcmp(model_type, "GoogleNet") == 0) {
        Info.numOfLayers    = 108;
        Info.ioMemCount     = 8394704;
        Info.filterMemCount = 107286016;
        Info.inputSize      = {3, 224, 224};
        Info.outputSize     = {1000};

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

    vector<int> size = {batchSize, inputSize[CHANNEL], inputSize[HEIGHT], inputSize[WIDTH]};

    if (strcmp(modelType, "LeNet") == 0) {
        LeNet(size);

    } else if (strcmp(modelType, "CaffeNet") == 0) {
        CaffeNet(size);

    } else if (strcmp(modelType, "ResNet18") == 0) {
        ResNet18(size);

    } else if (strcmp(modelType, "VGG16") == 0) {
        VGG16(size);

    } else if (strcmp(modelType, "GoogleNet") == 0) {
        GoogleNet(size);

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

    modelGraph->addLayer(new ResNetBlock18(layer_id++, modelGraph->getOFMapSize(), false));
    modelGraph->addLayer(new ResNetBlock18(layer_id++, modelGraph->getOFMapSize(), false));

    modelGraph->addLayer(new ResNetBlock18(layer_id++, modelGraph->getOFMapSize(), true));
    modelGraph->addLayer(new ResNetBlock18(layer_id++, modelGraph->getOFMapSize(), false));

    modelGraph->addLayer(new ResNetBlock18(layer_id++, modelGraph->getOFMapSize(), true));
    modelGraph->addLayer(new ResNetBlock18(layer_id++, modelGraph->getOFMapSize(), false));

    modelGraph->addLayer(new ResNetBlock18(layer_id++, modelGraph->getOFMapSize(), true));
    modelGraph->addLayer(new ResNetBlock18(layer_id++, modelGraph->getOFMapSize(), false));

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
    
    modelGraph->addLayer(new Inception(layer_id++, modelGraph->getOFMapSize(), 64, 96, 128, 16, 32, 32));
    modelGraph->addLayer(new Inception(layer_id++, modelGraph->getOFMapSize(), 128, 128, 192, 32, 96, 64));
    modelGraph->addLayer(new Pooling  (layer_id++, modelGraph->getOFMapSize(), {480, 480, 3, 3}, (char*)"Max_Pool", 2, 1));

    modelGraph->addLayer(new Inception(layer_id++, modelGraph->getOFMapSize(), 192, 96, 208, 16, 48, 64));
    modelGraph->addLayer(new Inception(layer_id++, modelGraph->getOFMapSize(), 160, 112, 224, 24, 64, 64));
    modelGraph->addLayer(new Inception(layer_id++, modelGraph->getOFMapSize(), 128, 128, 256, 24, 64, 64));
    modelGraph->addLayer(new Inception(layer_id++, modelGraph->getOFMapSize(), 112, 114, 288, 32, 64, 64));
    modelGraph->addLayer(new Inception(layer_id++, modelGraph->getOFMapSize(), 256, 160, 320, 32, 128, 128));
    modelGraph->addLayer(new Pooling  (layer_id++, modelGraph->getOFMapSize(), {832, 832, 3, 3}, (char*)"Max_Pool", 2, 1));

    modelGraph->addLayer(new Inception(layer_id++, modelGraph->getOFMapSize(), 256, 160, 320, 32, 128, 128));
    modelGraph->addLayer(new Inception(layer_id++, modelGraph->getOFMapSize(), 384, 192, 384, 48, 128, 128));

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