#include "include/App_config.h"
#include "include/Log.h"

#include "include/CPU.hpp"
#include "include/Layers.hpp"
#include "include/LayerGroup.hpp"
#include "include/MemoryControl.hpp"
#include "include/Models.hpp"


/* ************************************************************************************************
 * Main
 * ************************************************************************************************
 */
int main (int argc, char** argv)
{

    std::cout << "Hello GPGPU" << std::endl;
    
    MemoryControl mMC(DISK_SPACE, PAGE_SIZE);
    CPU mCPU(&mMC);
    // Model model;
    // BENCHMARK( model, ResNet18 );
    // model.printSummary();


    return 0;
}