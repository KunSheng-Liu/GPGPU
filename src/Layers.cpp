/**
 * \name    Layer.cpp
 * 
 * \brief   Implement the API
 * 
 * \date    Mar 31, 2023
 */

#include "include/Layers.hpp"

/* ************************************************************************************************
 * Global Variable
 * ************************************************************************************************
 */
int Layer::layerCount = 0;

/** ===============================================================================================
 * \name    Layer
 *
 * \brief   Construct a layer
 * 
 * \param   layer_type          the layer type
 * \param   input_size          [batch, channel, height, width]
 * \param   filter_size         [FILTER_CHANNEL_O, FILTER_CHANNEL_I, height, width]
 * \param   activation_type     the activation type
 * 
 * \endcond
 * ================================================================================================
 */
Layer::Layer(char* layer_type, vector<int>* input_size, vector<int>* filter_size, char* activation_type)
        : layerType(layer_type), iFMapSize(input_size), filterSize(filter_size), activationType(activation_type)
        , iFMap(nullptr), filter(nullptr), oFMapSize(nullptr), oFMap(nullptr), layerIndex(layerCount++), flagExecuting(false), flagFinish(false)
{
    if (iFMapSize  != nullptr)  iFMap = new vector<unsigned char> ((*iFMapSize)[BATCH]  * (*iFMapSize)[CHANNEL]  * (*iFMapSize)[HEIGHT]  * (*iFMapSize)[WIDTH]);
    if (filterSize != nullptr) filter = new vector<unsigned char> ((*filterSize)[BATCH] * (*filterSize)[CHANNEL] * (*filterSize)[HEIGHT] * (*filterSize)[WIDTH]);
}


/** ===============================================================================================
 * \name   ~Layer
 *
 * \brief   Destruct a layer
 * 
 * \endcond
 * ================================================================================================
 */
Layer::~Layer()
{

}


/** ===============================================================================================
 * \name    setIFMap
 *
 * \brief   Set the input feature map
 * 
 * \param   data        the pointer of vector
 * 
 * \endcond
 * ================================================================================================
 */
void
Layer::setIFMap(vector<unsigned char>* data)
{
    if (iFMap != nullptr) delete iFMap;
    iFMap  = data;
}


/** ===============================================================================================
 * \name    setFilter
 *
 * \brief   Set the output feature map
 * 
 * \param   data        the pointer of vector
 * 
 * \endcond
 * ================================================================================================
 */
void
Layer::setFilter(vector<unsigned char>* data)
{
    if (filter != nullptr) delete filter;
    filter = data;    
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
Layer::changeBatch(int new_batch_size)
{
    (*iFMapSize)[BATCH] = new_batch_size;
    (*oFMapSize)[BATCH] = new_batch_size;
    if (iFMap  != nullptr)  iFMap->resize((*iFMapSize)[BATCH]  * (*iFMapSize)[CHANNEL]  * (*iFMapSize)[HEIGHT]  * (*iFMapSize)[WIDTH]);
    if (oFMap  != nullptr)  oFMap->resize((*oFMapSize)[BATCH]  * (*oFMapSize)[CHANNEL]  * (*oFMapSize)[HEIGHT]  * (*oFMapSize)[WIDTH]);
}


/** ===============================================================================================
 * \name    memoryAllocate
 *
 * \brief   Allocate physical address to the model virtual address.
 * 
 * \param   mmu     the memory management unit
 * 
 * \endcond
 * ================================================================================================
 */
void
Layer::memoryAllocate(MMU* mmu)
{
    log_D("memoryAllocate", "ID: " + to_string(layerIndex) + "  " + layerType);
    if (LOG_LEVEL >= VERBOSE) cout << "iFMap ";
    if(iFMap)  mmu->memoryAllocate(static_cast<int>(reinterpret_cast<std::uintptr_t>(iFMap)),  iFMap->size()  * sizeof(unsigned char));
    if (LOG_LEVEL >= VERBOSE) cout << "oFMap ";
    if(oFMap)  mmu->memoryAllocate(static_cast<int>(reinterpret_cast<std::uintptr_t>(oFMap)),  oFMap->size()  * sizeof(unsigned char));
    if (LOG_LEVEL >= VERBOSE) cout << "filter ";
    if(filter) mmu->memoryAllocate(static_cast<int>(reinterpret_cast<std::uintptr_t>(filter)), filter->size() * sizeof(unsigned char));

}


/** ===============================================================================================
 * \name    compileToKernel
 *
 * \brief   Make the kernel dependency.
 * 
 * \param   mmu         the memory management unit
 * \param   container   the container to keep the compiled GPU requests
 * \param   dependency  the depended pointers of this kernel
 * 
 * \return  dependency of the next layer 
 * 
 * \endcond
 * ================================================================================================
 */
vector<Kernel*>
Layer::compileToKernel(vector<Kernel>& container, vector<Kernel*> dependency)
{
    log_D(layerType, "issueLayer");
    container.emplace_back();
    Kernel* kernelptr = &container.back();

    /* add config */
    kernelptr->srcLayer = this;
    kernelptr->kernelID = layerIndex;
    kernelptr->dependencyKernels = move(dependency);

    dependency.emplace_back(kernelptr);
    return move(dependency);
}


/** ===============================================================================================
 * \name    printInfo
 *
 * \brief   Print the layer information
 * 
 * \endcond
 * ================================================================================================
 */
void
Layer::printInfo()
{
    std::cout << "(" 
              << std::right << std::setw(4)  << (*iFMapSize)[BATCH]             << ", " \
              << std::right << std::setw(4)  << (*iFMapSize)[CHANNEL]           << ", " \
              << std::right << std::setw(3)  << (*iFMapSize)[HEIGHT]            << ", " \
              << std::right << std::setw(3)  << (*iFMapSize)[WIDTH]                     \
              << std::left  << std::setw(10) << ")" << "("                              \
              << std::right << std::setw(4)  << (*filterSize)[FILTER_CHANNEL_O] << ", " \
              << std::right << std::setw(4)  << (*filterSize)[FILTER_CHANNEL_I] << ", " \
              << std::right << std::setw(3)  << (*filterSize)[HEIGHT]           << ", " \
              << std::right << std::setw(3)  << (*filterSize)[WIDTH]                    \
              << std::left  << std::setw(10) << ")" << "("                              \
              << std::right << std::setw(4)  << (*oFMapSize)[BATCH]             << ", " \
              << std::right << std::setw(4)  << (*oFMapSize)[CHANNEL]           << ", " \
              << std::right << std::setw(3)  << (*oFMapSize)[HEIGHT]            << ", " \
              << std::right << std::setw(3)  << (*oFMapSize)[WIDTH]                     \
              << std::left  << std::setw(10) << ")"; 
}



/** ===============================================================================================
 * \name    Conv2D
 *
 * \brief   For the inheritance class
 * 
 * \param   layer_type          the layer type
 * \param   input_size          [batch, channel, height, width]
 * \param   filter_size         [FILTER_CHANNEL_O, FILTER_CHANNEL_I, height, width]
 * \param   activation_type     the activation type
 * \param   _stride             [height, width]
 * \param   _padding            [height, width]
 * 
 * \endcond
 * ================================================================================================
 */
Conv2D::Conv2D(char* layer_type, vector<int>* input_size, vector<int>* filter_size, char* activation_type, vector<int>* _stride, vector<int>* _padding)
        : Layer(layer_type, input_size, filter_size, activation_type)
        , stride(_stride), padding(_padding)
{
    calculateOFMapSize();
    int size = (*oFMapSize)[BATCH] * (*oFMapSize)[CHANNEL] * (*oFMapSize)[HEIGHT] * (*oFMapSize)[WIDTH];
    oFMap = new vector<unsigned char>(size);
}


/** ===============================================================================================
 * \name    Conv2D
 *
 * \brief   Construct a 2D convolution layer
 * 
 * \param   input_size          [batch, channel, height, width]
 * \param   filter_size         [FILTER_CHANNEL_O, FILTER_CHANNEL_I, height, width]
 * \param   activation_type     the activation type
 * \param   _stride             [height, width]
 * \param   _padding            [height, width]
 * 
 * \endcond
 * ================================================================================================
 */
Conv2D::Conv2D(vector<int>* input_size, vector<int>* filter_size, char* activation_type, vector<int>* _stride, vector<int>* _padding)
        : Conv2D((char*)"Conv2D", input_size, filter_size, activation_type, _stride, _padding)
{

}


/** ===============================================================================================
 * \name    Conv2D
 *
 * \brief   Construct a 2D convolution layer
 * 
 * \param   input_size          [batch, channel, height, width]
 * \param   filter_size         [FILTER_CHANNEL_O, FILTER_CHANNEL_I, height, width]
 * \param   activation_type     the activation type
 * \param   _stride             width
 * \param   _padding            width
 * 
 * \endcond
 * ================================================================================================
 */
Conv2D::Conv2D(vector<int>* input_size, vector<int>* filter_size, char* activation_type, int _stride, int _padding)
        : Conv2D((char*)"Conv2D", input_size, filter_size, activation_type, move(new vector<int>{_stride,_stride}), move(new vector<int>{_padding, _padding}))
{
    
}


/** ===============================================================================================
 * \name   ~Conv2D
 *
 * \brief   Destruct a 2D convolution layer
 * 
 * \endcond
 * ================================================================================================
 */
Conv2D::~Conv2D()
{
    
}


/** ===============================================================================================
 * \name    Conv2D
 *
 * \brief   Calculate the output feature map dimension
 * 
 * \endcond
 * ================================================================================================
 */
void
Conv2D::calculateOFMapSize()
{
    bool check = (!iFMapSize->empty()) && (!filterSize->empty()) && (!stride->empty()) && (!padding->empty());
    ASSERT(check, "Cannot calculate the size of OFMap due to missing parameter.");

    oFMapSize = new vector<int>();
    oFMapSize->emplace_back((*iFMapSize)[BATCH]);
    oFMapSize->emplace_back((*filterSize)[FILTER_CHANNEL_O]);
    oFMapSize->emplace_back((double) ((*iFMapSize)[HEIGHT] + 2 * (*padding)[STRIDE_PADDING_HEIGHT] - (*filterSize)[HEIGHT]) / (double) ((*stride)[STRIDE_PADDING_HEIGHT]) + 1);
    oFMapSize->emplace_back((double) ((*iFMapSize)[WIDTH]  + 2 * (*padding)[STRIDE_PADDING_WIDTH]  - (*filterSize)[WIDTH])  / (double) ((*stride)[STRIDE_PADDING_WIDTH])  + 1);
}


/** ===============================================================================================
 * \name    memoryAllocate
 *
 * \brief   Allocate physical address to the model virtual address.
 * 
 * \param   mmu     the memory management unit
 * 
 * \endcond
 * ================================================================================================
 */
// void
// Conv2D::memoryAllocate(MMU* mmu)
// {
//     Layer::memoryAllocate(mmu);
//     if (LOG_LEVEL >= VERBOSE) cout << "stride ";
//     if(stride)  mmu->memoryAllocate(static_cast<int>(reinterpret_cast<std::uintptr_t>(stride)),  stride->size()  * sizeof(unsigned char));
//     if (LOG_LEVEL >= VERBOSE) cout << "padding ";
//     if(padding)  mmu->memoryAllocate(static_cast<int>(reinterpret_cast<std::uintptr_t>(padding)),  padding->size()  * sizeof(unsigned char));

// }


/** ===============================================================================================
 * \name    printInfo
 *
 * \brief   Print the layer information
 * 
 * \endcond
 * ================================================================================================
 */
void
Conv2D::printInfo()
{
    std::cout << std::left << std::setw(10) << layerIndex; 
    std::cout << std::left << std::setw(16) << layerType; 
    std::cout << std::left << std::setw(13) << activationType;
    Layer::printInfo();
    std::cout << "(" << std::right << std::setw(2)  << (*stride)[STRIDE_PADDING_HEIGHT]  << ", " \
                     << std::right << std::setw(2)  << (*stride)[STRIDE_PADDING_WIDTH]           \
                     << std::left  << std::setw(10) << ")" << "("                                \
                     << std::right << std::setw(2)  << (*padding)[STRIDE_PADDING_HEIGHT] << ", " \
                     << std::right << std::setw(2)  << (*padding)[STRIDE_PADDING_WIDTH]  << ")";
                     
    std::cout << std::endl;
}


/** ===============================================================================================
 * \name    issueLayer
 *
 * \brief   Compile the layer into GPU requests.
 * 
 * \param   mmu         the memory management unit
 * \param   container   the container to keep the compiled GPU requests
 * \param   dependency  the depended pointers of this kernel
 * 
 * \return  dependency of the next layer
 * 
 * \endcond
 * ================================================================================================
 */
void 
Conv2D::issueLayer(MMU* mmu, Kernel* targetKernel)
{

    vector<int> iFMapPages   = mmu->addressTranslate(reinterpret_cast<std::uintptr_t>(iFMap));
    vector<int> oFMapPages   = mmu->addressTranslate(reinterpret_cast<std::uintptr_t>(oFMap));
    vector<int> filterPages  = mmu->addressTranslate(reinterpret_cast<std::uintptr_t>(filter));
    log_D("iFMapPages Num"  , to_string(iFMapPages.size()));
    log_D("oFMapPages Num"  , to_string(oFMapPages.size()));
    log_D("filterPages Num" , to_string(filterPages.size()));

    /* Use inverse order for let the address be closer */
    for (int w_o = 0; w_o < (*oFMapSize)[WIDTH]; w_o++)
    {
        for (int h_o = 0; h_o < (*oFMapSize)[HEIGHT]; h_o++)
        {
            for (int c_o = 0; c_o < (*oFMapSize)[CHANNEL]; c_o++)
            {
                for (int b = 0; b < (*oFMapSize)[BATCH]; b++)
                {                   
                    Request* request = new Request();

                    /* read filter pages */
                    const int f_start = c_o * (*filterSize)[FILTER_CHANNEL_I] * (*filterSize)[HEIGHT] * (*filterSize)[WIDTH] / PAGE_SIZE;
                    for (int f_offset = 0; f_offset <= (*filterSize)[FILTER_CHANNEL_I] * (*filterSize)[HEIGHT] * (*filterSize)[WIDTH] / PAGE_SIZE; f_offset++)
                    {
                        request->readPages.emplace_back(filterPages[f_start + f_offset]);
                    }

                    /* read input pages */
                    const int h_start = h_o * (*stride)[STRIDE_PADDING_HEIGHT] - (*padding)[STRIDE_PADDING_HEIGHT];
                    const int w_start = w_o * (*stride)[STRIDE_PADDING_HEIGHT] - (*padding)[STRIDE_PADDING_HEIGHT];

                    for (int c_i = 0; c_i < (*filterSize)[FILTER_CHANNEL_I]; c_i++)
                    {
                        for (int h_i = max(0, h_start); h_i < min(h_start + (*filterSize)[HEIGHT], (*iFMapSize)[HEIGHT]); h_i++)
                        {
                            int w_i = max(0, w_start);
                            const int i_start = b * (*iFMapSize)[CHANNEL] * (*iFMapSize)[HEIGHT] * (*iFMapSize)[WIDTH] + c_i * (*iFMapSize)[HEIGHT] * (*iFMapSize)[WIDTH] + h_i * (*iFMapSize)[WIDTH] + w_i;
                            
                            if (request->readPages.back() != iFMapPages[i_start / PAGE_SIZE]) 
                            {
                                request->readPages.emplace_back(iFMapPages[i_start / PAGE_SIZE]);
                            }

                            if (PAGE_SIZE - i_start % PAGE_SIZE > (*iFMapSize)[WIDTH] - 1)
                                continue;
                            else
                                request->readPages.emplace_back((i_start / PAGE_SIZE) + 1);
                        }
                    }

                    // cout << "[" << b << ", " << c_o << ", " << h_o << ", " << w_o << "] Read Pages: ";
                    // for(int page : request->readPages)
                    // {
                    //     cout << page << ", ";
                    // }
                    // cout << endl;

                    /* write ouput page */
                    request->writePages.emplace_back(oFMapPages[b * c_o * h_o * w_o / PAGE_SIZE]);

                    /* for the activation exectuion */
                    if (strcmp(activationType, "None") != 0)
                        request->numOfInstructions++;  // for the activation exectuion

                    targetKernel->addRequest(move(request));
                }
                
            }
            
        }
        
    }

    log_D("Num of request", to_string(targetKernel->requests.size()));
    log_D("Num of read address", to_string(targetKernel->numOfRead));
    log_D("Num of write address", to_string(targetKernel->numOfWrite));

}



/** ===============================================================================================
 * \name    Pooling
 *
 * \brief   Construct a pooling layer
 * 
 * \param   input_size          [batch, channel, height, width]
 * \param   filter_size         [FILTER_CHANNEL_O, FILTER_CHANNEL_I, height, width]
 * \param   activation_type     the activation type
 * \param   _stride             [height, width]
 * \param   _padding            [height, width]
 * 
 * \endcond
 * ================================================================================================
 */
Pooling::Pooling(vector<int>* input_size, vector<int>* filter_size, char* activation_type, vector<int>* _stride, vector<int>* _padding)
        : Conv2D((char*)"Pooling", input_size, filter_size, activation_type, _stride, _padding)
{
    
}


/** ===============================================================================================
 * \name    Pooling
 *
 * \brief   Construct a pooling layer
 * 
 * \param   input_size          [batch, channel, height, width]
 * \param   filter_size         [FILTER_CHANNEL_O, FILTER_CHANNEL_I, height, width]
 * \param   activation_type     the activation type
 * \param   _stride             [height, width]
 * \param   _padding            [height, width]
 * 
 * \endcond
 * ================================================================================================
 */
Pooling::Pooling(vector<int>* input_size, vector<int>* filter_size, char* activation_type, int _stride, int _padding)
        : Conv2D((char*)"Pooling", input_size, filter_size, activation_type, move(new vector<int>{_stride,_stride}), move(new vector<int>{_padding, _padding}))
{
    
}


/** ===============================================================================================
 * \name    issueLayer
 *
 * \brief   Compile the layer into GPU requests.
 * 
 * \param   mmu         the memory management unit
 * \param   container   the container to keep the compiled GPU requests
 * \param   dependency  the depended pointers of this kernel
 * 
 * \endcond
 * ================================================================================================
 */
void 
Pooling::issueLayer(MMU* mmu, Kernel* targetKernel)
{

}



/** ===============================================================================================
 * \name    Flatten
 *
 * \brief   Construct a flatten layer
 * 
 * \param   input_size          [batch, channel, height, width]
 * 
 * \endcond
 * ================================================================================================
 */
Flatten::Flatten(vector<int>* input_size)
        : Layer((char*)"Flatten", input_size)
{
    calculateOFMapSize();
    int size = (*oFMapSize)[BATCH] * (*oFMapSize)[CHANNEL] * (*oFMapSize)[HEIGHT] * (*oFMapSize)[WIDTH];
    oFMap = new vector<unsigned char>(size);
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
Flatten::calculateOFMapSize()
{
    bool check = (!iFMapSize->empty());
    ASSERT(check, "Cannot calculate the size of OFMap due to missing parameter.");

    oFMapSize = new vector<int>();
    (*oFMapSize).emplace_back((*iFMapSize)[BATCH]);
    (*oFMapSize).emplace_back((*iFMapSize)[CHANNEL] * (*iFMapSize)[HEIGHT] * (*iFMapSize)[WIDTH]);
    (*oFMapSize).emplace_back(1);
    (*oFMapSize).emplace_back(1);
}


/** ===============================================================================================
 * \name    printInfo
 *
 * \brief   Print the layer information
 * 
 * \endcond
 * ================================================================================================
 */
void
Flatten::printInfo()
{

    std::cout << std::left << std::setw(10) << layerIndex \
              << std::left << std::setw(16) << layerType  \
              << std::left << std::setw(13) << activationType;

    std::cout << "(" 
              << std::right << std::setw(4)  << (*iFMapSize)[BATCH]   << ", " \
              << std::right << std::setw(4)  << (*iFMapSize)[CHANNEL] << ", " \
              << std::right << std::setw(3)  << (*iFMapSize)[HEIGHT]  << ", " \
              << std::right << std::setw(3)  << (*iFMapSize)[WIDTH]           \
              << std::left  << std::setw(10) << ")"         
              << std::right << std::setw(22) << "None"                        \
              << std::right << setw(10) << "("
              << std::right << std::setw(4)  << (*oFMapSize)[BATCH]   << ", " \
              << std::right << std::setw(4)  << (*oFMapSize)[CHANNEL] << ", " \
              << std::right << std::setw(3)  << (*oFMapSize)[HEIGHT]  << ", " \
              << std::right << std::setw(3)  << (*oFMapSize)[WIDTH]           \
              << std::left  << std::setw(10) << ")";  

    std::cout << std::endl;
}


/** ===============================================================================================
 * \name    issueLayer
 *
 * \brief   Compile the layer into GPU requests.
 * 
 * \param   mmu         the memory management unit
 * \param   container   the container to keep the compiled GPU requests
 * \param   dependency  the depended pointers of this kernel
 * 
 * \endcond
 * ================================================================================================
 */
void 
Flatten::issueLayer(MMU* mmu, Kernel* targetKernel)
{

}



/** ===============================================================================================
 * \name    ByPass
 *
 * \brief   Construct a flatten layer
 * 
 * \param   input_size          [batch, channel, height, width]
 * 
 * \endcond
 * ================================================================================================
 */
ByPass::ByPass(vector<int>* input_size)
        : Layer((char*)"ByPass", input_size)
{
    calculateOFMapSize();
    int size = (*oFMapSize)[BATCH] * (*oFMapSize)[CHANNEL] * (*oFMapSize)[HEIGHT] * (*oFMapSize)[WIDTH];
    oFMap = new vector<unsigned char>(size);
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
ByPass::calculateOFMapSize()
{
    bool check = (!iFMapSize->empty());
    ASSERT(check, "Cannot calculate the size of OFMap due to missing parameter.");
    oFMapSize = iFMapSize;
}


/** ===============================================================================================
 * \name    printInfo
 *
 * \brief   Print the layer information
 * 
 * \endcond
 * ================================================================================================
 */
void
ByPass::printInfo()
{

    std::cout << std::left << std::setw(10) << layerIndex \
              << std::left << std::setw(16) << layerType  \
              << std::left << std::setw(13) << activationType;

    std::cout << "(" 
              << std::right << std::setw(4)  << (*iFMapSize)[BATCH]   << ", " \
              << std::right << std::setw(4)  << (*iFMapSize)[CHANNEL] << ", " \
              << std::right << std::setw(3)  << (*iFMapSize)[HEIGHT]  << ", " \
              << std::right << std::setw(3)  << (*iFMapSize)[WIDTH]           \
              << std::left  << std::setw(10) << ")"  
              << std::right << std::setw(22) << "None"                        \
              << std::right << setw(10) << "("
              << std::right << std::setw(4)  << (*oFMapSize)[BATCH]   << ", " \
              << std::right << std::setw(4)  << (*oFMapSize)[CHANNEL] << ", " \
              << std::right << std::setw(3)  << (*oFMapSize)[HEIGHT]  << ", " \
              << std::right << std::setw(3)  << (*oFMapSize)[WIDTH]           \
              << std::left  << std::setw(10) << ")";  

    std::cout << std::endl;
}


/** ===============================================================================================
 * \name    issueLayer
 *
 * \brief   Compile the layer into GPU requests.
 * 
 * \param   mmu         the memory management unit
 * \param   container   the container to keep the compiled GPU requests
 * \param   dependency  the depended pointers of this kernel
 * 
 * \endcond
 * ================================================================================================
 */
void 
ByPass::issueLayer(MMU* mmu, Kernel* targetKernel)
{

}



/** ===============================================================================================
 * \name    Dense
 *
 * \brief   Construct a dense layer
 * 
 * \param   input_size          [batch, channel, height, width]
 * \param   filter_size         [FILTER_CHANNEL_O, FILTER_CHANNEL_I, height, width]
 * \param   activation_type     the activation type
 * 
 * \endcond
 * ================================================================================================
 */
Dense::Dense(vector<int>* input_size, vector<int>* filter_size, char* activation_type)
        : Layer((char*)"Dense", input_size, filter_size, activation_type)
{
    calculateOFMapSize();
}


/** ===============================================================================================
 * \name    Dense
 *
 * \brief   Construct a dense layer
 * 
 * \param   input_size          [batch, channel, height, width]
 * \param   output_width        the number of output feature
 * 
 * \endcond
 * ================================================================================================
 */
Dense::Dense(vector<int>* input_size, int output_width)
        : Layer((char*)"Dense", input_size), outWidth(output_width)
{
    calculateOFMapSize();
    int size = (*oFMapSize)[BATCH] * (*oFMapSize)[CHANNEL] * (*oFMapSize)[HEIGHT] * (*oFMapSize)[WIDTH];
    oFMap = new vector<unsigned char>(size);
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
Dense::calculateOFMapSize()
{
    bool check = (!iFMapSize->empty());
    ASSERT(check, "Cannot calculate the size of OFMap due to missing parameter.");

    oFMapSize = new vector<int>();
    (*oFMapSize).emplace_back((*iFMapSize)[BATCH]);
    (*oFMapSize).emplace_back(outWidth);
    (*oFMapSize).emplace_back(1);
    (*oFMapSize).emplace_back(1);
}


/** ===============================================================================================
 * \name    printInfo
 *
 * \brief   Print the layer information
 * 
 * \endcond
 * ================================================================================================
 */
void
Dense::printInfo()
{

    std::cout << std::left << std::setw(10) << layerIndex \
              << std::left << std::setw(16) << layerType  \
              << std::left << std::setw(13) << activationType;

    std::cout << "(" 
              << std::right << std::setw(4)  << (*iFMapSize)[BATCH]   << ", " \
              << std::right << std::setw(4)  << (*iFMapSize)[CHANNEL] << ", " \
              << std::right << std::setw(3)  << (*iFMapSize)[HEIGHT]  << ", " \
              << std::right << std::setw(3)  << (*iFMapSize)[WIDTH]           \
              << std::left  << std::setw(10) << ")"
              << std::right << std::setw(22) << "None"                        \
              << std::right << setw(10) << "("                                \
              << std::right << std::setw(4)  << (*oFMapSize)[BATCH]   << ", " \
              << std::right << std::setw(4)  << (*oFMapSize)[CHANNEL] << ", " \
              << std::right << std::setw(3)  << (*oFMapSize)[HEIGHT]  << ", " \
              << std::right << std::setw(3)  << (*oFMapSize)[WIDTH]           \
              << std::left  << std::setw(10) << ")"; 

    std::cout << std::endl;
}


/** ===============================================================================================
 * \name    issueLayer
 *
 * \brief   Compile the layer into GPU requests.
 * 
 * \param   mmu         the memory management unit
 * \param   container   the container to keep the compiled GPU requests
 * \param   dependency  the depended pointers of this kernel
 * 
 * \endcond
 * ================================================================================================
 */
void 
Dense::issueLayer(MMU* mmu, Kernel* targetKernel)
{

}

