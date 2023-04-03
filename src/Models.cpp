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
 *   \ /
 *    4        add               64        56 x 56                           ReLU
 *   / \
 *  |   5   Conv2D    3 x 3      64        56 x 56     1          1          ReLU
 *  |   6   Conv2D    3 x 3      64        56 x 56     1          1             
 *   \ /
 *    7        add               64        56 x 56                           ReLU
 *   / \
 *  |   8   Conv2D    3 x 3     128        28 x 28     2          1          ReLU
 *  |   9   Conv2D    3 x 3     128        28 x 28     1          1           
 *   \ /
 *   10        add              128        28 x 28                           ReLU
 *   / \
 *  |  11   Conv2D    3 x 3     128        28 x 28     1          1          ReLU
 *  |  12   Conv2D    3 x 3     128        28 x 28     1          1           
 *   \ /
 *   13        add              128        28 x 28                           ReLU
 *   / \
 *  |  14   Conv2D    3 x 3     256        14 x 14     2          1          ReLU
 *  |  15   Conv2D    3 x 3     256        14 x 14     1          1           
 *   \ /
 *   16        add              256        14 x 14                           ReLU
 *   / \
 *  |  14   Conv2D    3 x 3     256        14 x 14     1          1          ReLU
 *  |  15   Conv2D    3 x 3     256        14 x 14     1          1           
 *   \ /
 *   16        add              256        14 x 14                           ReLU
 *   / \
 *  |  14   Conv2D    3 x 3     512         7 x 7      2          1          ReLU
 *  |  15   Conv2D    3 x 3     512         7 x 7      1          1           
 *   \ /
 *   16        add              512        14 x 14                           ReLU
 *   / \
 *  |  14   Conv2D    3 x 3     512         7 x 7      1          1          ReLU
 *  |  15   Conv2D    3 x 3     512         7 x 7      1          1           
 *   \ /
 *   16        add              512         7 x 7                            ReLU
 *   17       Pool    7 x 7    1024         1 x 1      2          1          ReLU
 *   20      Dense    1 x 1    1000         1 x 1
 * ================================================================================================
 */
void
Model::ResNet18()
{
    modelName = (char*)"ResNet18";
    LayerGroup resnet18;

    resnet18.addLayer(new Conv2D (new int[4]{1, 3, 224, 224}, new int[4]{3, 64, 7, 7}, (char*)"ReLU", 2, 3));
    resnet18.addLayer(new Pooling(new int[4]{1, 64, 112, 112}, new int[4]{64, 64, 3, 3}, (char*)"None", 2, 1));

    LayerGroup sequential_0;
    LayerGroup branch_0(Group_t::CaseCode); 

    branch_0.addLayer(new Conv2D(new int[4]{1, 64, 56, 56}, new int[4]{64, 256, 1, 1}, (char*)"ReLU", 1, 0));

    sequential_0.addLayer(new Conv2D(new int[4]{1, 64, 56, 56}, new int[4]{64, 64, 1, 1},  (char*)"ReLU", 1, 0));
    sequential_0.addLayer(new Conv2D(new int[4]{1, 64, 56, 56}, new int[4]{64, 64, 3, 3},  (char*)"ReLU", 1, 1));
    sequential_0.addLayer(new Conv2D(new int[4]{1, 64, 56, 56}, new int[4]{64, 256, 1, 1}, (char*)"ReLU", 1, 0));
    branch_0.addLayer(&sequential_0);

    resnet18.addLayer(&branch_0);

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