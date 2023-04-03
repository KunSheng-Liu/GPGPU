#include "include/App_config.h"
#include "include/Log.h"
#include "include/Layers.hpp"
#include "include/LayerGroup.hpp"
#include "include/Models.hpp"

#define BENCHMARK( obj, model ) obj.model()

/* ************************************************************************************************
 * Main
 * ************************************************************************************************
 */
int main (int argc, char** argv)
{

    std::cout << "Hello GPGPU" << std::endl;
    
    Model model;
    BENCHMARK( model, VGG16 );
    model.printSummary();

    return 0;
}