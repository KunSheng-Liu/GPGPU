/**
 * \name    Models.cpp
 * 
 * \brief   Implement the API
 * 
 * \date    Mar 31, 2023
 */

#include "include/Models.hpp"

/* ************************************************************************************************
 * Global Variable
 * ************************************************************************************************
 */
int Model::ModelCount = 0;


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
Model::Model(): modelIndex(++ModelCount)
{
    layerGroup = new LayerGroup();
}


/** ===============================================================================================
 * \name   ~Model
 * 
 * \brief   Construct a model
 * 
 * \param   name    the model name
 * 
 * \endcond
 * ================================================================================================
 */
Model::~Model()
{
    delete layerGroup;
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
 *    0     Conv2D    3 x 3      64       224 x 224    1          1          ReLU
 *    1     Conv2D    3 x 3      64       224 x 224    1          1          ReLU
 *    2       Pool    2 x 2      64       112 x 112    2          0
 *    3     Conv2D    3 x 3     128       112 x 112    1          1          ReLU
 *    4     Conv2D    3 x 3     128       112 x 112    1          1          ReLU
 *    5       Pool    2 x 2     128        56 x 56     2          0
 *    6     Conv2D    3 x 3     256        56 x 56     1          1          ReLU
 *    7     Conv2D    3 x 3     256        56 x 56     1          1          ReLU
 *    8     Conv2D    3 x 3     256        56 x 56     1          1          ReLU
 *    9       Pool    2 x 2     256        28 x 28     2          0
 *   10     Conv2D    3 x 3     512        28 x 28     1          1          ReLU
 *   11     Conv2D    3 x 3     512        28 x 28     1          1          ReLU
 *   12     Conv2D    3 x 3     512        28 x 28     1          1          ReLU
 *   13       Pool    2 x 2     512        14 x 14     2          0
 *   14     Conv2D    3 x 3     512        14 x 14     1          1          ReLU
 *   15     Conv2D    3 x 3     512        14 x 14     1          1          ReLU
 *   16     Conv2D    3 x 3     512        14 x 14     1          1          ReLU
 *   17       Pool    2 x 2     512         7 x 7      2          0
 *         Flatten            25088         1 x 1
 *   18      Dense    1 x 1    4096         1 x 1                            ReLU
 *   19      Dense    1 x 1    4096         1 x 1                            ReLU 
 *   20      Dense    1 x 1    1000         1 x 1    
 * 
 * ================================================================================================
 */
void
Model::VGG16()
{
    modelName = (char*)"VGG16";
    
    layerGroup->addLayer(new Conv2D (new int[4]{1,  3, 224, 224}, new int[4]{ 1, 64, 3, 3}, (char*)"ReLU", 1, 1));
    layerGroup->addLayer(new Conv2D (new int[4]{1, 64, 224, 224}, new int[4]{64, 64, 3, 3}, (char*)"ReLU", 1, 1));
    layerGroup->addLayer(new Pooling(new int[4]{1, 64, 224, 224}, new int[4]{64, 64, 2, 2}, (char*)"None", 2, 0));
    
    layerGroup->addLayer(new Conv2D (new int[4]{1,  64, 112, 112}, new int[4]{64,  128, 3, 3}, (char*)"ReLU", 1, 1));
    layerGroup->addLayer(new Conv2D (new int[4]{1, 128, 112, 112}, new int[4]{128, 128, 3, 3}, (char*)"ReLU", 1, 1));
    layerGroup->addLayer(new Pooling(new int[4]{1, 128, 112, 112}, new int[4]{128, 128, 2, 2}, (char*)"None", 2, 0));

    layerGroup->addLayer(new Conv2D (new int[4]{1, 128, 56, 56}, new int[4]{128, 256, 3, 3}, (char*)"ReLU", 1, 1));
    layerGroup->addLayer(new Conv2D (new int[4]{1, 256, 56, 56}, new int[4]{256, 256, 3, 3}, (char*)"ReLU", 1, 1));
    layerGroup->addLayer(new Conv2D (new int[4]{1, 256, 56, 56}, new int[4]{256, 256, 3, 3}, (char*)"ReLU", 1, 1));
    layerGroup->addLayer(new Pooling(new int[4]{1, 256, 56, 56}, new int[4]{256, 256, 2, 2}, (char*)"None", 2, 0));

    layerGroup->addLayer(new Conv2D (new int[4]{1, 256, 28, 28}, new int[4]{256, 512, 3, 3}, (char*)"ReLU", 1, 1));
    layerGroup->addLayer(new Conv2D (new int[4]{1, 512, 28, 28}, new int[4]{512, 512, 3, 3}, (char*)"ReLU", 1, 1));
    layerGroup->addLayer(new Conv2D (new int[4]{1, 512, 28, 28}, new int[4]{512, 512, 3, 3}, (char*)"ReLU", 1, 1));
    layerGroup->addLayer(new Pooling(new int[4]{1, 512, 28, 28}, new int[4]{512, 512, 2, 2}, (char*)"None", 2, 0));

    layerGroup->addLayer(new Conv2D (new int[4]{1, 512, 14, 14}, new int[4]{512, 512, 3, 3}, (char*)"ReLU", 1, 1));
    layerGroup->addLayer(new Conv2D (new int[4]{1, 512, 14, 14}, new int[4]{512, 512, 3, 3}, (char*)"ReLU", 1, 1));
    layerGroup->addLayer(new Conv2D (new int[4]{1, 512, 14, 14}, new int[4]{512, 512, 3, 3}, (char*)"ReLU", 1, 1));
    layerGroup->addLayer(new Pooling(new int[4]{1, 512, 14, 14}, new int[4]{512, 512, 2, 2}, (char*)"None", 2, 0));

    layerGroup->addLayer(new Flatten(new int[4]{1, 512, 7, 7}));

    layerGroup->addLayer(new Dense(new int[4]{1, 25088, 1, 1}, 4096));
    layerGroup->addLayer(new Dense(new int[4]{1,  4096, 1, 1}, 4096));
    layerGroup->addLayer(new Dense(new int[4]{1,  4096, 1, 1}, 1000));

    numOfLayer  =  22;
    inputLayer  = *layerGroup->layers.begin();
    outputLayer = *layerGroup->layers.end();
    
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
 *    0     Conv2D    7 x 7      64       112 x 112    2          3          ReLU
 *    1       Pool    3 x 3      64        56 x 56     2          1          
 *   / \
 *  |   2   Conv2D    3 x 3      64        56 x 56     1          1          ReLU
 *  |   3   Conv2D    3 x 3      64        56 x 56     1          1        
 *  4   |   ByPass               64        56 x 56
 *   \ /
 *    |
 *   / \
 *  |   5   Conv2D    3 x 3      64        56 x 56     1          1          ReLU
 *  |   6   Conv2D    3 x 3      64        56 x 56     1          1             
 *  7   |   ByPass               64        56 x 56
 *   \ /
 *    |
 *   / \
 *  |   8   Conv2D    3 x 3     128        28 x 28     2          1          ReLU
 *  |   9   Conv2D    3 x 3     128        28 x 28     1          1           
 * 10   |   Conv2D    3 x 3     128        28 x 28     2          1          ReLU
 *   \ /
 *    |        
 *   / \
 *  |  11   Conv2D    3 x 3     128        28 x 28     1          1          ReLU
 *  |  12   Conv2D    3 x 3     128        28 x 28     1          1               
 * 13   |   ByPass              128        28 x 28
 *   \ /
 *    |
 *   / \
 *  |  14   Conv2D    3 x 3     256        14 x 14     2          1          ReLU
 *  |  15   Conv2D    3 x 3     256        14 x 14     1          1           
 * 16   |   Conv2D    3 x 3     256        14 x 14     2          1          ReLU
 *   \ /
 *    | 
 *   / \
 *  |  17   Conv2D    3 x 3     256        14 x 14     1          1          ReLU
 *  |  18   Conv2D    3 x 3     256        14 x 14     1          1                    
 * 19   |   ByPass              256        14 x 14
 *   \ /
 *    |  
 *   / \
 *  |  20   Conv2D    3 x 3     512         7 x 7      2          1          ReLU
 *  |  21   Conv2D    3 x 3     512         7 x 7      1          1       
 * 22   |   Conv2D    3 x 3     512         7 x 7      2          1          ReLU    
 *   \ /
 *    |
 *   / \
 *  |  23   Conv2D    3 x 3     512         7 x 7      1          1          ReLU
 *  |  24   Conv2D    3 x 3     512         7 x 7      1          1                             
 * 25   |   ByPass              512         7 x 7 
 *   \ /
 *    |
 *   26       Pool    7 x 7    1024         1 x 1      2          1          ReLU
 *   27      Dense    1 x 1    1000         1 x 1
 * ================================================================================================
 */
void
Model::ResNet18()
{
    modelName = (char*)"ResNet18";
    LayerGroup resnet18;

    layerGroup->addLayer(new Conv2D (new int[4]{1, 3, 224, 224}, new int[4]{3, 64, 7, 7}, (char*)"ReLU", 2, 3));
    layerGroup->addLayer(new Pooling(new int[4]{1, 64, 112, 112}, new int[4]{64, 64, 3, 3}, (char*)"None", 2, 1));

    /* Stage 1_1 */
    LayerGroup* sequential_1_1 = new LayerGroup();
    LayerGroup* branch_1_1     = new LayerGroup(Group_t::CaseCode); 

    sequential_1_1->addLayer(new Conv2D(new int[4]{1, 64, 56, 56}, new int[4]{64, 64, 3, 3},  (char*)"ReLU", 1, 1));
    sequential_1_1->addLayer(new Conv2D(new int[4]{1, 64, 56, 56}, new int[4]{64, 64, 3, 3},  (char*)"ReLU", 1, 1));
    
    branch_1_1->addLayer(sequential_1_1);
    branch_1_1->addLayer(new ByPass(new int[4]{1, 64, 56, 56}));

    layerGroup->addLayer(branch_1_1);


    /* Stage 1_2 */
    LayerGroup* sequential_1_2 = new LayerGroup();
    LayerGroup* branch_1_2 = new LayerGroup(Group_t::CaseCode); 

    sequential_1_2->addLayer(new Conv2D(new int[4]{1, 64, 56, 56}, new int[4]{64, 64, 3, 3},  (char*)"ReLU", 1, 1));
    sequential_1_2->addLayer(new Conv2D(new int[4]{1, 64, 56, 56}, new int[4]{64, 64, 3, 3},  (char*)"ReLU", 1, 1));
    
    branch_1_2->addLayer(sequential_1_2);
    branch_1_2->addLayer(new ByPass(new int[4]{1, 64, 56, 56}));

    layerGroup->addLayer(branch_1_2);

    
    /* Stage 2_1 */
    LayerGroup* sequential_2_1 = new LayerGroup();
    LayerGroup* branch_2_1 = new LayerGroup(Group_t::CaseCode); 

    sequential_2_1->addLayer(new Conv2D(new int[4]{1,  64, 56, 56}, new int[4]{ 64, 128, 3, 3},  (char*)"ReLU", 2, 1));
    sequential_2_1->addLayer(new Conv2D(new int[4]{1, 128, 28, 28}, new int[4]{128, 128, 3, 3},  (char*)"ReLU", 1, 1));
    
    branch_2_1->addLayer(sequential_2_1);
    branch_2_1->addLayer(new Conv2D(new int[4]{1, 64, 56, 56}, new int[4]{64, 128, 3, 3},  (char*)"ReLU", 2, 1));

    layerGroup->addLayer(branch_2_1);
    
    /* Stage 2_2 */
    LayerGroup* sequential_2_2 = new LayerGroup();
    LayerGroup* branch_2_2 = new LayerGroup(Group_t::CaseCode); 

    sequential_2_2->addLayer(new Conv2D(new int[4]{1, 128, 28, 28}, new int[4]{128, 128, 3, 3},  (char*)"ReLU", 1, 1));
    sequential_2_2->addLayer(new Conv2D(new int[4]{1, 128, 28, 28}, new int[4]{128, 128, 3, 3},  (char*)"ReLU", 1, 1));
    
    branch_2_2->addLayer(sequential_2_2);
    branch_2_2->addLayer(new ByPass(new int[4]{1, 128, 28, 28}));

    layerGroup->addLayer(branch_2_2);


    /* Stage 3_1 */
    LayerGroup* sequential_3_1 = new LayerGroup();
    LayerGroup* branch_3_1 = new LayerGroup(Group_t::CaseCode); 

    sequential_3_1->addLayer(new Conv2D(new int[4]{1, 128, 28, 28}, new int[4]{128, 256, 3, 3},  (char*)"ReLU", 2, 1));
    sequential_3_1->addLayer(new Conv2D(new int[4]{1, 256, 14, 14}, new int[4]{256, 256, 3, 3},  (char*)"ReLU", 1, 1));
    
    branch_3_1->addLayer(sequential_3_1);
    branch_3_1->addLayer(new Conv2D(new int[4]{1, 128, 28, 28}, new int[4]{128, 256, 3, 3},  (char*)"ReLU", 2, 1));

    layerGroup->addLayer(branch_3_1);
    
    /* Stage 3_2 */
    LayerGroup* sequential_3_2 = new LayerGroup();
    LayerGroup* branch_3_2 = new LayerGroup(Group_t::CaseCode); 

    sequential_3_2->addLayer(new Conv2D(new int[4]{1, 256, 14, 14}, new int[4]{256, 256, 3, 3},  (char*)"ReLU", 1, 1));
    sequential_3_2->addLayer(new Conv2D(new int[4]{1, 256, 14, 14}, new int[4]{256, 256, 3, 3},  (char*)"ReLU", 1, 1));
    
    branch_3_2->addLayer(sequential_3_2);
    branch_3_2->addLayer(new ByPass(new int[4]{1, 256, 14, 14}));

    layerGroup->addLayer(branch_3_2);


    /* Stage _4_1 */
    LayerGroup* sequential__4_1 = new LayerGroup();
    LayerGroup* branch__4_1 = new LayerGroup(Group_t::CaseCode); 

    sequential__4_1->addLayer(new Conv2D(new int[4]{1, 256, 14, 14}, new int[4]{256, 512, 3, 3},  (char*)"ReLU", 2, 1));
    sequential__4_1->addLayer(new Conv2D(new int[4]{1, 512, 7, 7}, new int[4]{512, 512, 3, 3},  (char*)"ReLU", 1, 1));
    
    branch__4_1->addLayer(sequential__4_1);
    branch__4_1->addLayer(new Conv2D(new int[4]{1, 256, 14, 14}, new int[4]{256, 512, 3, 3},  (char*)"ReLU", 2, 1));

    layerGroup->addLayer(branch__4_1);
    
    /* Stage _4_2 */
    LayerGroup* sequential__4_2 = new LayerGroup();
    LayerGroup* branch__4_2 = new LayerGroup(Group_t::CaseCode); 

    sequential__4_2->addLayer(new Conv2D(new int[4]{1, 512, 7, 7}, new int[4]{512, 512, 3, 3},  (char*)"ReLU", 1, 1));
    sequential__4_2->addLayer(new Conv2D(new int[4]{1, 512, 7, 7}, new int[4]{512, 512, 3, 3},  (char*)"ReLU", 1, 1));
    
    branch__4_2->addLayer(sequential__4_2);
    branch__4_2->addLayer(new ByPass(new int[4]{1, 512, 7, 7}));

    layerGroup->addLayer(branch__4_2);

    layerGroup->addLayer(new Pooling(new int[4]{1, 512, 7, 7}, new int[4]{512, 1024, 7, 7}, (char*)"None", 2, 0));
    layerGroup->addLayer(new Dense(new int[4]{1, 1024, 1, 1}, 1000));

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
    std::cout << modelName << " summary:" << endl;
    layerGroup->printInfo();
}