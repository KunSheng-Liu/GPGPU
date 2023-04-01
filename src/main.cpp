#include "include/App_config.h"
#include "include/Log.h"
#include "include/Layers.hpp"
#include "include/LayerGroup.hpp"

/* ************************************************************************************************
 * Main
 * ************************************************************************************************
 */
int main (int argc, char** argv)
{

    std::cout << "Hello GPGPU" << std::endl;
    
    
    LayerGroup vgg16(Group_t::CaseCade);
    vgg16.addLayer(new Conv2D (new int[4]{1,  3, 224, 224}, new int[4]{ 1, 64, 3, 3}, (char*)"ReLU", 1, 1));
    vgg16.addLayer(new Conv2D (new int[4]{1, 64, 224, 224}, new int[4]{64, 64, 3, 3}, (char*)"ReLU", 1, 1));
    vgg16.addLayer(new Pooling(new int[4]{1, 64, 224, 224}, new int[4]{64, 64, 2, 2}, (char*)"None", 2, 0));
    
    vgg16.addLayer(new Conv2D (new int[4]{1,  64, 112, 112}, new int[4]{64,  128, 3, 3}, (char*)"ReLU", 1, 1));
    vgg16.addLayer(new Conv2D (new int[4]{1, 128, 112, 112}, new int[4]{128, 128, 3, 3}, (char*)"ReLU", 1, 1));
    vgg16.addLayer(new Pooling(new int[4]{1, 128, 112, 112}, new int[4]{128, 128, 2, 2}, (char*)"None", 2, 0));

    vgg16.addLayer(new Conv2D (new int[4]{1, 128, 56, 56}, new int[4]{128, 256, 3, 3}, (char*)"ReLU", 1, 1));
    vgg16.addLayer(new Conv2D (new int[4]{1, 256, 56, 56}, new int[4]{256, 256, 3, 3}, (char*)"ReLU", 1, 1));
    vgg16.addLayer(new Conv2D (new int[4]{1, 256, 56, 56}, new int[4]{256, 256, 3, 3}, (char*)"ReLU", 1, 1));
    vgg16.addLayer(new Pooling(new int[4]{1, 256, 56, 56}, new int[4]{256, 256, 2, 2}, (char*)"None", 2, 0));

    vgg16.addLayer(new Conv2D (new int[4]{1, 256, 28, 28}, new int[4]{256, 512, 3, 3}, (char*)"ReLU", 1, 1));
    vgg16.addLayer(new Conv2D (new int[4]{1, 512, 28, 28}, new int[4]{512, 512, 3, 3}, (char*)"ReLU", 1, 1));
    vgg16.addLayer(new Conv2D (new int[4]{1, 512, 28, 28}, new int[4]{512, 512, 3, 3}, (char*)"ReLU", 1, 1));
    vgg16.addLayer(new Pooling(new int[4]{1, 512, 28, 28}, new int[4]{512, 512, 2, 2}, (char*)"None", 2, 0));

    vgg16.addLayer(new Conv2D (new int[4]{1, 512, 14, 14}, new int[4]{512, 512, 3, 3}, (char*)"ReLU", 1, 1));
    vgg16.addLayer(new Conv2D (new int[4]{1, 512, 14, 14}, new int[4]{512, 512, 3, 3}, (char*)"ReLU", 1, 1));
    vgg16.addLayer(new Conv2D (new int[4]{1, 512, 14, 14}, new int[4]{512, 512, 3, 3}, (char*)"ReLU", 1, 1));
    vgg16.addLayer(new Pooling(new int[4]{1, 512, 14, 14}, new int[4]{512, 512, 2, 2}, (char*)"None", 2, 0));

    vgg16.addLayer(new Flatten(new int[4]{1, 512, 7, 7}));

    vgg16.addLayer(new Dense(new int[4]{1, 25088, 1, 1}, 4096));
    vgg16.addLayer(new Dense(new int[4]{1,  4096, 1, 1}, 4096));
    vgg16.addLayer(new Dense(new int[4]{1,  4096, 1, 1}, 1000));

    vgg16.printInfo();

    return 0;
}