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
 * \endcond
 * ================================================================================================
 */
Model::Model(int app_id): appID(app_id), modelID(modelCount++)
{
    modelGraph = new LayerGroup();
}


/** ===============================================================================================
 * \name    Model
 * 
 * \brief   Construct a model
 * 
 * \param   name    the model name
 * 
 * \endcond
 * ================================================================================================
 */
Model::Model(int app_id, int batch_size): appID(app_id), modelID(modelCount++), batchSize(batch_size)
{
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
    for (auto& kernel : kernelContainer)
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
    bool finish = true;
    for (auto& kernel : kernelContainer)
    {
        finish &= kernel.isFinish();
    }
    return finish;
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

    if (strcmp(model_type, "Test") == 0) {
        Info.numOfLayers    = 3;
        Info.ioMemCount     = 7375872;
        Info.filterMemCount = 54976;
        Info.inputSize      = { 3, 224, 224};
        Info.outputSize     = {64, 112, 112};

    } else if (strcmp(model_type, "VGG16") == 0) {
        Info.numOfLayers    = 22;
        Info.ioMemCount     = 15262696;
        Info.filterMemCount = 17151680;
        Info.inputSize      = {3, 224, 224};
        Info.inputSize      = {1000};

    } else if (strcmp(model_type, "ResNet18") == 0) {
        Info.numOfLayers    = 28;
        Info.ioMemCount     = 4166632;
        Info.filterMemCount = 38270144;
        Info.inputSize      = {3, 224, 224};
        Info.inputSize      = {1000};

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
Model::buildLayerGraph(const char* model_type)
{
    log_T("Model", "buildLayerGraph");

    if (strcmp(model_type, "Test") == 0) {
        Test();

    } else if (strcmp(model_type, "VGG16") == 0) {
        VGG16();

    } else if (strcmp(model_type, "ResNet18") == 0) {
        ResNet18();
    }
}

/** ===============================================================================================
 * \name    None
 * 
 * \brief   Build the None layer graph
 * 
 * \endcond
 * 
 * #Index    Type    Kernel    Feature     Output    Stride    Padding    Activation
 *                    Size       Map        Size
 *            Data               512      224 x 224 
 *    1     Conv2D    3 x 3      512       14 x 14     1          1          ReLU
 *    2       Pool    2 x 2      512        7 x 7      2          0
 *    3    Flatten             25088        1 x 1
 *    4      Dense    1 x 1     1000        1 x 1                            ReLU
 * ================================================================================================
 */
void 
Model::Test()
{
    modelName = (char*)"Test";
    
    modelGraph->addLayer(new Conv2D (new vector<int>{batchSize,   3, 14, 14}, new vector<int>{512,   3, 3, 3}, (char*)"ReLU", 1, 1));
    modelGraph->addLayer(new Pooling(new vector<int>{batchSize, 512, 14, 14}, new vector<int>{512, 512, 2, 2}, (char*)"None", 2, 0));
                                                      
    modelGraph->addLayer(new Flatten(new vector<int>{batchSize, 512, 7, 7}));
                                                        
    modelGraph->addLayer(new Dense(new vector<int>{batchSize, 25088, 1, 1}, 1000));

    numOfLayer = 4;
    
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
Model::VGG16()
{
    modelName = (char*)"VGG16";
    
    modelGraph->addLayer(new Conv2D (new vector<int>{batchSize,  3, 224, 224}, new vector<int>{64,  3, 3, 3}, (char*)"ReLU", 1, 1));
    modelGraph->addLayer(new Conv2D (new vector<int>{batchSize, 64, 224, 224}, new vector<int>{64, 64, 3, 3}, (char*)"ReLU", 1, 1));
    modelGraph->addLayer(new Pooling(new vector<int>{batchSize, 64, 224, 224}, new vector<int>{64, 64, 2, 2}, (char*)"None", 2, 0));
                                                      
    modelGraph->addLayer(new Conv2D (new vector<int>{batchSize,  64, 112, 112}, new vector<int>{128,  64, 3, 3}, (char*)"ReLU", 1, 1));
    modelGraph->addLayer(new Conv2D (new vector<int>{batchSize, 128, 112, 112}, new vector<int>{128, 128, 3, 3}, (char*)"ReLU", 1, 1));
    modelGraph->addLayer(new Pooling(new vector<int>{batchSize, 128, 112, 112}, new vector<int>{128, 128, 2, 2}, (char*)"None", 2, 0));
                                                      
    modelGraph->addLayer(new Conv2D (new vector<int>{batchSize, 128, 56, 56}, new vector<int>{256, 128, 3, 3}, (char*)"ReLU", 1, 1));
    modelGraph->addLayer(new Conv2D (new vector<int>{batchSize, 256, 56, 56}, new vector<int>{256, 256, 3, 3}, (char*)"ReLU", 1, 1));
    modelGraph->addLayer(new Conv2D (new vector<int>{batchSize, 256, 56, 56}, new vector<int>{256, 256, 3, 3}, (char*)"ReLU", 1, 1));
    modelGraph->addLayer(new Pooling(new vector<int>{batchSize, 256, 56, 56}, new vector<int>{256, 256, 2, 2}, (char*)"None", 2, 0));
                                                      
    modelGraph->addLayer(new Conv2D (new vector<int>{batchSize, 256, 28, 28}, new vector<int>{512, 256, 3, 3}, (char*)"ReLU", 1, 1));
    modelGraph->addLayer(new Conv2D (new vector<int>{batchSize, 512, 28, 28}, new vector<int>{512, 512, 3, 3}, (char*)"ReLU", 1, 1));
    modelGraph->addLayer(new Conv2D (new vector<int>{batchSize, 512, 28, 28}, new vector<int>{512, 512, 3, 3}, (char*)"ReLU", 1, 1));
    modelGraph->addLayer(new Pooling(new vector<int>{batchSize, 512, 28, 28}, new vector<int>{512, 512, 2, 2}, (char*)"None", 2, 0));
                                                      
    modelGraph->addLayer(new Conv2D (new vector<int>{batchSize, 512, 14, 14}, new vector<int>{512, 512, 3, 3}, (char*)"ReLU", 1, 1));
    modelGraph->addLayer(new Conv2D (new vector<int>{batchSize, 512, 14, 14}, new vector<int>{512, 512, 3, 3}, (char*)"ReLU", 1, 1));
    modelGraph->addLayer(new Conv2D (new vector<int>{batchSize, 512, 14, 14}, new vector<int>{512, 512, 3, 3}, (char*)"ReLU", 1, 1));
    modelGraph->addLayer(new Pooling(new vector<int>{batchSize, 512, 14, 14}, new vector<int>{512, 512, 2, 2}, (char*)"None", 2, 0));
                                                      
    modelGraph->addLayer(new Flatten(new vector<int>{batchSize, 512, 7, 7}));
                                                        
    modelGraph->addLayer(new Dense(new vector<int>{batchSize, 25088, 1, 1}, 4096));
    modelGraph->addLayer(new Dense(new vector<int>{batchSize,  4096, 1, 1}, 4096));
    modelGraph->addLayer(new Dense(new vector<int>{batchSize,  4096, 1, 1}, 1000));

    numOfLayer = 22;
    
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
 * \endcond
 * 
 * #Index    Type    Kernel    Feature     Output    Stride    Padding    Activation
 *                    Size       Map        Size
 *            Data                3       224 x 224
 *    1     Conv2D    7 x 7      64       112 x 112    2          3          ReLU
 *    2       Pool    3 x 3      64        56 x 56     2          1          
 *   / \
 *  |   3   Conv2D    3 x 3      64        56 x 56     1          1          ReLU
 *  |   4   Conv2D    3 x 3      64        56 x 56     1          1        
 *  5   |   ByPass               64        56 x 56
 *   \ /
 *    |
 *   / \
 *  |   6   Conv2D    3 x 3      64        56 x 56     1          1          ReLU
 *  |   7   Conv2D    3 x 3      64        56 x 56     1          1             
 *  8   |   ByPass               64        56 x 56
 *   \ /
 *    |
 *   / \
 *  |   9   Conv2D    3 x 3     128        28 x 28     2          1          ReLU
 *  |  10   Conv2D    3 x 3     128        28 x 28     1          1           
 * 11   |   Conv2D    3 x 3     128        28 x 28     2          1          ReLU
 *   \ /
 *    |        
 *   / \
 *  |  12   Conv2D    3 x 3     128        28 x 28     1          1          ReLU
 *  |  13   Conv2D    3 x 3     128        28 x 28     1          1               
 * 14   |   ByPass              128        28 x 28
 *   \ /
 *    |
 *   / \
 *  |  15   Conv2D    3 x 3     256        14 x 14     2          1          ReLU
 *  |  16   Conv2D    3 x 3     256        14 x 14     1          1           
 * 17   |   Conv2D    3 x 3     256        14 x 14     2          1          ReLU
 *   \ /
 *    | 
 *   / \
 *  |  18   Conv2D    3 x 3     256        14 x 14     1          1          ReLU
 *  |  19   Conv2D    3 x 3     256        14 x 14     1          1                    
 * 20   |   ByPass              256        14 x 14
 *   \ /
 *    |  
 *   / \
 *  |  21   Conv2D    3 x 3     512         7 x 7      2          1          ReLU
 *  |  22   Conv2D    3 x 3     512         7 x 7      1          1       
 * 23   |   Conv2D    3 x 3     512         7 x 7      2          1          ReLU    
 *   \ /
 *    |
 *   / \
 *  |  24   Conv2D    3 x 3     512         7 x 7      1          1          ReLU
 *  |  25   Conv2D    3 x 3     512         7 x 7      1          1                             
 * 26   |   ByPass              512         7 x 7 
 *   \ /
 *    |
 *   27       Pool    7 x 7    1024         1 x 1      2          1          ReLU
 *   28      Dense    1 x 1    1000         1 x 1
 * ================================================================================================
 */
void
Model::ResNet18()
{
    modelName = (char*)"ResNet18";

    modelGraph->addLayer(new Conv2D (new vector<int>{batchSize, 3, 224, 224} , new vector<int>{64,  3, 7, 7}, (char*)"ReLU", 2, 3));
    modelGraph->addLayer(new Pooling(new vector<int>{batchSize, 64, 112, 112}, new vector<int>{64, 64, 3, 3}, (char*)"None", 2, 1));

    /* Stage 1_1 */
    LayerGroup* sequential_1_1 = new LayerGroup();
    LayerGroup* branch_1_1     = new LayerGroup(Group_t::CaseCode); 

    sequential_1_1->addLayer(new Conv2D(new vector<int>{batchSize, 64, 56, 56}, new vector<int>{64, 64, 3, 3},  (char*)"ReLU", 1, 1));
    sequential_1_1->addLayer(new Conv2D(new vector<int>{batchSize, 64, 56, 56}, new vector<int>{64, 64, 3, 3},  (char*)"ReLU", 1, 1));
    
    branch_1_1->addLayer(sequential_1_1);
    branch_1_1->addLayer(new ByPass(new vector<int>{batchSize, 64, 56, 56}));

    modelGraph->addLayer(branch_1_1);


    /* Stage 1_2 */
    LayerGroup* sequential_1_2 = new LayerGroup();
    LayerGroup* branch_1_2 = new LayerGroup(Group_t::CaseCode); 

    sequential_1_2->addLayer(new Conv2D(new vector<int>{batchSize, 64, 56, 56}, new vector<int>{64, 64, 3, 3},  (char*)"ReLU", 1, 1));
    sequential_1_2->addLayer(new Conv2D(new vector<int>{batchSize, 64, 56, 56}, new vector<int>{64, 64, 3, 3},  (char*)"ReLU", 1, 1));
    
    branch_1_2->addLayer(sequential_1_2);
    branch_1_2->addLayer(new ByPass(new vector<int>{batchSize, 64, 56, 56}));

    modelGraph->addLayer(branch_1_2);

    
    /* Stage 2_1 */
    LayerGroup* sequential_2_1 = new LayerGroup();
    LayerGroup* branch_2_1 = new LayerGroup(Group_t::CaseCode); 

    sequential_2_1->addLayer(new Conv2D(new vector<int>{batchSize,  64, 56, 56}, new vector<int>{128,  64, 3, 3},  (char*)"ReLU", 2, 1));
    sequential_2_1->addLayer(new Conv2D(new vector<int>{batchSize, 128, 28, 28}, new vector<int>{128, 128, 3, 3},  (char*)"ReLU", 1, 1));
    
    branch_2_1->addLayer(sequential_2_1);
    branch_2_1->addLayer(new Conv2D(new vector<int>{batchSize, 64, 56, 56}, new vector<int>{128, 64, 3, 3},  (char*)"ReLU", 2, 1));

    modelGraph->addLayer(branch_2_1);
    
    /* Stage 2_2 */
    LayerGroup* sequential_2_2 = new LayerGroup();
    LayerGroup* branch_2_2 = new LayerGroup(Group_t::CaseCode); 

    sequential_2_2->addLayer(new Conv2D(new vector<int>{batchSize, 128, 28, 28}, new vector<int>{128, 128, 3, 3},  (char*)"ReLU", 1, 1));
    sequential_2_2->addLayer(new Conv2D(new vector<int>{batchSize, 128, 28, 28}, new vector<int>{128, 128, 3, 3},  (char*)"ReLU", 1, 1));
    
    branch_2_2->addLayer(sequential_2_2);
    branch_2_2->addLayer(new ByPass(new vector<int>{batchSize, 128, 28, 28}));

    modelGraph->addLayer(branch_2_2);


    /* Stage 3_1 */
    LayerGroup* sequential_3_1 = new LayerGroup();
    LayerGroup* branch_3_1 = new LayerGroup(Group_t::CaseCode); 

    sequential_3_1->addLayer(new Conv2D(new vector<int>{batchSize, 128, 28, 28}, new vector<int>{256, 128, 3, 3},  (char*)"ReLU", 2, 1));
    sequential_3_1->addLayer(new Conv2D(new vector<int>{batchSize, 256, 14, 14}, new vector<int>{256, 256, 3, 3},  (char*)"ReLU", 1, 1));
    
    branch_3_1->addLayer(sequential_3_1);
    branch_3_1->addLayer(new Conv2D(new vector<int>{batchSize, 128, 28, 28}, new vector<int>{256, 128, 3, 3},  (char*)"ReLU", 2, 1));

    modelGraph->addLayer(branch_3_1);
    
    /* Stage 3_2 */
    LayerGroup* sequential_3_2 = new LayerGroup();
    LayerGroup* branch_3_2 = new LayerGroup(Group_t::CaseCode); 

    sequential_3_2->addLayer(new Conv2D(new vector<int>{batchSize, 256, 14, 14}, new vector<int>{256, 256, 3, 3},  (char*)"ReLU", 1, 1));
    sequential_3_2->addLayer(new Conv2D(new vector<int>{batchSize, 256, 14, 14}, new vector<int>{256, 256, 3, 3},  (char*)"ReLU", 1, 1));
    
    branch_3_2->addLayer(sequential_3_2);
    branch_3_2->addLayer(new ByPass(new vector<int>{batchSize, 256, 14, 14}));

    modelGraph->addLayer(branch_3_2);


    /* Stage _4_1 */
    LayerGroup* sequential__4_1 = new LayerGroup();
    LayerGroup* branch__4_1 = new LayerGroup(Group_t::CaseCode); 

    sequential__4_1->addLayer(new Conv2D(new vector<int>{batchSize, 256, 14, 14}, new vector<int>{512, 256, 3, 3},  (char*)"ReLU", 2, 1));
    sequential__4_1->addLayer(new Conv2D(new vector<int>{batchSize, 512, 7, 7}, new vector<int>{512, 512, 3, 3},  (char*)"ReLU", 1, 1));
    
    branch__4_1->addLayer(sequential__4_1);
    branch__4_1->addLayer(new Conv2D(new vector<int>{batchSize, 256, 14, 14}, new vector<int>{512, 256, 3, 3},  (char*)"ReLU", 2, 1));

    modelGraph->addLayer(branch__4_1);
    
    /* Stage _4_2 */
    LayerGroup* sequential__4_2 = new LayerGroup();
    LayerGroup* branch__4_2 = new LayerGroup(Group_t::CaseCode); 

    sequential__4_2->addLayer(new Conv2D(new vector<int>{batchSize, 512, 7, 7}, new vector<int>{512, 512, 3, 3},  (char*)"ReLU", 1, 1));
    sequential__4_2->addLayer(new Conv2D(new vector<int>{batchSize, 512, 7, 7}, new vector<int>{512, 512, 3, 3},  (char*)"ReLU", 1, 1));
    
    branch__4_2->addLayer(sequential__4_2);
    branch__4_2->addLayer(new ByPass(new vector<int>{batchSize, 512, 7, 7}));

    modelGraph->addLayer(branch__4_2);

    modelGraph->addLayer(new Pooling(new vector<int>{batchSize, 512, 7, 7}, new vector<int>{1024, 512, 7, 7}, (char*)"None", 2, 0));
    modelGraph->addLayer(new Dense(new vector<int>{batchSize, 1024, 1, 1}, 1000));

    numOfLayer = 28;

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
    std::cout << "Model " << modelID << ", " << modelName << " summary:" << std::endl;
    std::cout << std::left << std::setw(10) << "Layer_ID"    \
              << std::left << std::setw(12) << "Layer_Type"  \
              << std::left << std::setw(25)  << "Activation_Type" \
              << std::left << std::setw(30)  << "Input_Size" \
              << std::left << std::setw(30)  << "Filter_Size" \
              << std::left << std::setw(26)  << "Output_Size" \
              << std::left << std::setw(17)  << "Stride" \
              << std::left << std::setw(10)  << "Padding" ;

    std::cout << std::endl;

    modelGraph->printInfo();
    
}