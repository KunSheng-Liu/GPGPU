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
int Layer::vaCount = 0;

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
 * \note    use char* to store the layer_type is because it much easier to print out
 * 
 * \endcond
 * ================================================================================================
 */
Layer::Layer(int layer_id, char* layer_type, vector<int> input_size, vector<int> filter_size, char* activation_type)
        : layerType(layer_type), iFMapSize(input_size), filterSize(filter_size), activationType(activation_type)
        , iFMap({}), filter({}), oFMapSize({}), oFMap({}), layerID(layer_id)
{
    if (!iFMapSize.empty())  iFMap  = make_pair(++vaCount, new vector<DATA_TYPE> (iFMapSize[BATCH]  * iFMapSize[CHANNEL]  * iFMapSize[HEIGHT]  * iFMapSize[WIDTH]));
    if (!filterSize.empty()) filter = make_pair(++vaCount, new vector<DATA_TYPE> (filterSize[BATCH] * filterSize[CHANNEL] * filterSize[HEIGHT] * filterSize[WIDTH]));
}


/** ===============================================================================================
 * \name   ~Layer
 *
 * \brief   Destruct a layer
 * 
 * \note    Only release the pointer that without sharing. Should custom release the pointer by the 
 *          user after using.
 * 
 * \endcond
 * ================================================================================================
 */
Layer::~Layer()
{
    log_V("~Layer()", layerType);
    if (layerID == -1) return;

    if (oFMap.second)  delete oFMap.second;
    // if (filter.second) delete filter.second;
}


/** ===============================================================================================
 * \name    release
 * 
 * \brief   Release no used memory space
 * 
 * \return  The recorder contains memory access information
 * 
 * \note    Not release the oFMap because it will become next layer iFMap and release in next layer
 *  
 * \endcond
 * ================================================================================================
 */
PageRecord
Layer::memoryRelease(MMU* mmu)
{
    PageRecord record;
    record += mmu->getPageSummary(iFMap.first);
    record += mmu->getPageSummary(oFMap.first);
    record += mmu->getPageSummary(filter.first);

    mmu->memoryRelease(iFMap.first);
    mmu->memoryRelease(oFMap.first);
    mmu->memoryRelease(filter.first);

    if (iFMap.second)
    {
        iFMap.second->clear();
        iFMap.second->shrink_to_fit();
    }
    if (filter.second)
    {
        filter.second->clear();
        filter.second->shrink_to_fit();  
    }

    return record;
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
Layer::setIFMap(pair<int, vector<DATA_TYPE>*> data)
{
    if (iFMap.second) delete iFMap.second;
    iFMap = data;
}


/** ===============================================================================================
 * \name    setOFMap
 *
 * \brief   Set the output feature map
 * 
 * \param   data        the pointer of vector
 * 
 * \endcond
 * ================================================================================================
 */
void
Layer::setOFMap(pair<int, vector<DATA_TYPE>*> data)
{
    oFMap = data;
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
Layer::setFilter(pair<int, vector<DATA_TYPE>*> data)
{
    if (filter.second) delete filter.second;
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
    iFMapSize[BATCH] = new_batch_size;
    oFMapSize[BATCH] = new_batch_size;
    if (iFMap.second != nullptr)  iFMap.second->resize(iFMapSize[BATCH] * iFMapSize[CHANNEL] * iFMapSize[HEIGHT] * iFMapSize[WIDTH]);
    if (oFMap.second != nullptr)  oFMap.second->resize(oFMapSize[BATCH] * oFMapSize[CHANNEL] * oFMapSize[HEIGHT] * oFMapSize[WIDTH]);
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
    log_V("memoryAllocate", "ID: " + to_string(layerID) + "  " + layerType);
    if (PRINT_MEMORY_ALLOCATION) log("iFMap",  "", Color::Cyan);
    if(iFMap.second)  mmu->memoryAllocate(iFMap.first,  iFMap.second->size()  * sizeof(DATA_TYPE));
    if (PRINT_MEMORY_ALLOCATION) log("oFMap",  "", Color::Cyan);
    if(oFMap.second)  mmu->memoryAllocate(oFMap.first,  oFMap.second->size()  * sizeof(DATA_TYPE));
    if (PRINT_MEMORY_ALLOCATION) log("filter", "", Color::Cyan);
    if(filter.second) mmu->memoryAllocate(filter.first, filter.second->size() * sizeof(DATA_TYPE));

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
Layer::compileToKernel(int app_id, int model_id, vector<Kernel>& container, vector<Kernel*> dependency)
{
    container.emplace_back(Kernel(app_id, model_id, this, move(dependency)));

    dependency.emplace_back(&container.back());
    return move(dependency);
}


/** ===============================================================================================
 * \name    getMemoryUsage
 *
 * \brief   get the number of pages used in this layer
 * 
 * \return  int
 * 
 * \endcond
 * ================================================================================================
 */
unsigned long long
Layer::getMemoryUsage()
{
    unsigned long long  usage = 0;
    if(iFMap.second)  usage += iFMap.second->size()  * sizeof(DATA_TYPE);
    if(oFMap.second)  usage += oFMap.second->size()  * sizeof(DATA_TYPE);
    if(filter.second) usage += filter.second->size() * sizeof(DATA_TYPE);

    return usage;
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
              << std::right << std::setw(4)  << iFMapSize[BATCH]             << ", " \
              << std::right << std::setw(4)  << iFMapSize[CHANNEL]           << ", " \
              << std::right << std::setw(3)  << iFMapSize[HEIGHT]            << ", " \
              << std::right << std::setw(3)  << iFMapSize[WIDTH]                     \
              << std::left  << std::setw(10) << ")" << "("                              \
              << std::right << std::setw(4)  << filterSize[FILTER_CHANNEL_O] << ", " \
              << std::right << std::setw(4)  << filterSize[FILTER_CHANNEL_I] << ", " \
              << std::right << std::setw(3)  << filterSize[HEIGHT]           << ", " \
              << std::right << std::setw(3)  << filterSize[WIDTH]                    \
              << std::left  << std::setw(10) << ")" << "("                              \
              << std::right << std::setw(4)  << oFMapSize[BATCH]             << ", " \
              << std::right << std::setw(4)  << oFMapSize[CHANNEL]           << ", " \
              << std::right << std::setw(3)  << oFMapSize[HEIGHT]            << ", " \
              << std::right << std::setw(3)  << oFMapSize[WIDTH]                     \
              << std::left  << std::setw(10) << ")"; 
}


/** ===============================================================================================
 * \name    Compile
 *
 * \brief   Compile the layer into request
 * 
 * \param   mmu         the memory management unit
 * \param   container   the container to keep the compiled GPU requests
 * 
 * \endcond
 * ================================================================================================
 */
const int THREAD_NUM = 8;
/*
 * ================================================================================================
 */
void
Layer::Compile(MMU* mmu, Kernel* targetKernel)
{
    log_V("Compiling kernel", to_string(targetKernel->kernelID) + " ...");

    int numThread = ENABLE_THREAD_COMPILE ? THREAD_NUM : 1;
    pthread_t threads[numThread];
    vector<queue<Request*>> requestQueues(numThread);

    for (int i = 0; i < numThread; i++)
    {
        pthread_create(&threads[i], NULL, threadCompile, move(new ThreadArg(i, numThread, this, mmu, &requestQueues[i])));
    }
    
    for (int i = 0; i < numThread; i++)
    {
        pthread_join(threads[i], NULL);
    }

    for (auto& queue : requestQueues)
    {
        while(!queue.empty())
        {
            targetKernel->addRequest(move(queue.front()));
            queue.pop();
        }
    }

    Kernel::KernelInfo info = targetKernel->getKernelInfo();
    log_T("Num of request", to_string(info.numOfRequest));
    log_T("Num of read address", to_string(info.numOfRead));
    log_T("Num of write address", to_string(info.numOfWrite));
}


/** ===============================================================================================
 * \name    threadCompile
 *
 * \brief   A thread wrapper
 * 
 * \param   arg     the void* of threadArg
 * 
 * \endcond
 * ================================================================================================
 */
void* 
Layer::threadCompile(void* arg)
{
    ThreadArg* threadArg = static_cast<ThreadArg*>(arg);

#if (LOG_LEVEL >= VERBOSE)
    pthread_mutex_lock ( ioMutex );
        log_T(threadArg->srcLayer->layerType, "issueLayer");
    pthread_mutex_unlock ( ioMutex );
#endif
    threadArg->srcLayer->issueLayer(move(threadArg));

    pthread_exit(nullptr);
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
Conv2D::Conv2D(int layer_id, char* layer_type, vector<int> input_size, vector<int> filter_size, char* activation_type, vector<int> _stride, vector<int> _padding)
        : Layer(layer_id, layer_type, input_size, filter_size, activation_type)
        , stride(_stride), padding(_padding)
{
    calculateOFMapSize();
    int size = oFMapSize[BATCH] * oFMapSize[CHANNEL] * oFMapSize[HEIGHT] * oFMapSize[WIDTH];
    oFMap = make_pair(++vaCount, new vector<DATA_TYPE>(size));
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
Conv2D::Conv2D(int layer_id, vector<int> input_size, vector<int> filter_size, char* activation_type, vector<int> _stride, vector<int> _padding)
        : Conv2D(layer_id, (char*)"Conv2D", input_size, filter_size, activation_type, _stride, _padding)
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
Conv2D::Conv2D(int layer_id, vector<int> input_size, vector<int> filter_size, char* activation_type, int _stride, int _padding)
        : Conv2D(layer_id, (char*)"Conv2D", input_size, filter_size, activation_type, {_stride,_stride}, {_padding, _padding})
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
    bool check = !iFMapSize.empty() && !filterSize.empty() && !stride.empty() && !padding.empty();
    ASSERT(check, "Cannot calculate the size of OFMap due to missing parameter.");

    oFMapSize.emplace_back(iFMapSize[BATCH]);
    oFMapSize.emplace_back(filterSize[FILTER_CHANNEL_O]);
    oFMapSize.emplace_back((double) (iFMapSize[HEIGHT] + 2 * padding[STRIDE_PADDING_HEIGHT] - filterSize[HEIGHT]) / (double) stride[STRIDE_PADDING_HEIGHT] + 1);
    oFMapSize.emplace_back((double) (iFMapSize[WIDTH]  + 2 * padding[STRIDE_PADDING_WIDTH]  - filterSize[WIDTH])  / (double) stride[STRIDE_PADDING_WIDTH]  + 1);
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
//     if (LOG_LEVEL >= VERBOSE) std::cout << "stride ";
//     if(stride)  mmu->memoryAllocate(stride)),  stride->size()  * sizeof(DATA_TYPE));
//     if (LOG_LEVEL >= VERBOSE) std::cout << "padding ";
//     if(padding)  mmu->memoryAllocate(padding)),  padding->size()  * sizeof(DATA_TYPE));

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
    std::cout << std::left << std::setw(10) << layerID; 
    std::cout << std::left << std::setw(16) << layerType; 
    std::cout << std::left << std::setw(13) << activationType;
    Layer::printInfo();
    std::cout << "(" << std::right << std::setw(2)  << stride[STRIDE_PADDING_HEIGHT]  << ", "
                     << std::right << std::setw(2)  << stride[STRIDE_PADDING_WIDTH]   
                     << std::left  << std::setw(10) << ")" << "("
                     << std::right << std::setw(2)  << padding[STRIDE_PADDING_HEIGHT] << ", "
                     << std::right << std::setw(2)  << padding[STRIDE_PADDING_WIDTH]  << ")";
                     
    std::cout << std::endl;
}


/** ===============================================================================================
 * \name    issueLayer
 *
 * \brief   Conv2D perfrom the element multiplication on iFMap to filter at each place
 * 
 * \note    The batch size must be designed
 * 
 * \note    The memory address must be allocated
 * 
 * \param   arg     the void* of threadArg
 * 
 * \endcond
 * ================================================================================================
 */
void 
Conv2D::issueLayer(ThreadArg* threadArg)
{
    MMU* mmu = threadArg->mmu;

    pthread_mutex_lock ( ioMutex );
        vector<unsigned long long> iFMapPages   = mmu->addressTranslate(iFMap.first);
        vector<unsigned long long> oFMapPages   = mmu->addressTranslate(oFMap.first);
        vector<unsigned long long> filterPages  = mmu->addressTranslate(filter.first);
        log_V("iFMapPages Num"  , to_string(iFMapPages.size()));
        log_V("oFMapPages Num"  , to_string(oFMapPages.size()));
        log_V("filterPages Num" , to_string(filterPages.size()));
    pthread_mutex_unlock ( ioMutex );

    int data_byte = sizeof(DATA_TYPE);

    /* Thread compile start and end index */
    int start_index = (oFMapSize[WIDTH] * threadArg->threadID) / threadArg->numThread;
    int end_index   = (oFMapSize[WIDTH] * (threadArg->threadID + 1)) / threadArg->numThread;

    /* Use inverse order for let the address be closer */
    int index = 0;
    for (int w_o = start_index; w_o < end_index; w_o++)
    {
        for (int h_o = 0; h_o < oFMapSize[HEIGHT]; h_o++)
        {
            for (int c_o = 0; c_o < oFMapSize[CHANNEL]; c_o++)
            {
                for (int b = 0; b < oFMapSize[BATCH]; b++)
                {                   
                    Request* request = new Request();

                    const int h_start = h_o * stride[STRIDE_PADDING_HEIGHT] - padding[STRIDE_PADDING_HEIGHT];
                    const int w_start = w_o * stride[STRIDE_PADDING_HEIGHT] - padding[STRIDE_PADDING_HEIGHT];

                    for (int c_i = 0; c_i < filterSize[FILTER_CHANNEL_I]; c_i++)
                    {
                        for (int h_i = max(0, h_start); h_i < min(h_start + filterSize[HEIGHT], iFMapSize[HEIGHT]); h_i++)
                        {
                            for (int w_i = max(0, w_start); w_i < min(w_start + filterSize[WIDTH], iFMapSize[WIDTH]); w_i++)
                            {
                                /* read filter pages */
                                index = floor((c_i * filterSize[HEIGHT] * filterSize[WIDTH] + h_i * filterSize[WIDTH] + w_i) * data_byte / PAGE_SIZE);
                                ASSERT(index < filterPages.size(), "Layer " + to_string(layerID) + " (" + layerType + ") Overflow!");

                                request->readPages.emplace_back(make_pair(filterPages[index], 1));
                                
                                /* read input pages */
                                index = floor((b * iFMapSize[CHANNEL] * iFMapSize[HEIGHT] * iFMapSize[WIDTH] + c_i * iFMapSize[HEIGHT] * iFMapSize[WIDTH] + h_i * iFMapSize[WIDTH] + w_i) * data_byte / PAGE_SIZE);
                                ASSERT(index < iFMapPages.size(), "Layer " + to_string(layerID) + " (" + layerType + ") Overflow!");

                                request->readPages.emplace_back(make_pair(iFMapPages[index], 1));
                            }                 
                        }
                    }
                    
                    // Conv2D perfrom the element multiplication on iFMap to filter at each place
                    request->numOfInstructions = filterSize[HEIGHT] * filterSize[WIDTH];

                    /* write result to pages */
                    index = floor((b * oFMapSize[CHANNEL] * oFMapSize[HEIGHT] * oFMapSize[WIDTH] + c_o * oFMapSize[HEIGHT] * oFMapSize[WIDTH] + h_o * oFMapSize[WIDTH] + w_o) * data_byte / PAGE_SIZE);
                    ASSERT(index < oFMapPages.size(), "Layer " + to_string(layerID) + " (" + layerType + ") Overflow!");

                    request->writePages.emplace_back(make_pair(oFMapPages[floor((b * oFMapSize[CHANNEL] * oFMapSize[HEIGHT] * oFMapSize[WIDTH] + c_o * oFMapSize[HEIGHT] * oFMapSize[WIDTH] + h_o * oFMapSize[WIDTH] + w_o) * data_byte / PAGE_SIZE)], data_byte));

                    /* for the activation exectuion */
                    if (strcmp(activationType, "None") != 0) request->numOfInstructions++;  // for the activation exectuion

                    threadArg->requestQueue->push(move(Kernel::compressRequest(request)));
                }
                
            }
            
        }
        
    }

    delete threadArg;
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
Pooling::Pooling(int layer_id, vector<int> input_size, vector<int> filter_size, char* activation_type, vector<int> _stride, vector<int> _padding)
        : Conv2D(layer_id, (char*)"Pooling", input_size, filter_size, activation_type, _stride, _padding)
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
Pooling::Pooling(int layer_id, vector<int> input_size, vector<int> filter_size, char* activation_type, int _stride, int _padding)
        : Conv2D(layer_id, (char*)"Pooling", input_size, filter_size, activation_type, {_stride,_stride}, {_padding, _padding})
{
    
}


/** ===============================================================================================
 * \name    issueLayer
 *
 * \brief   Pooling layer find the maxinum / mininum input data in the field masked by filter
 * 
 * \note    The batch size must be designed
 * 
 * \note    The memory address must be allocated
 * 
 * \param   arg     the void* of threadArg
 * 
 * \endcond
 * ================================================================================================
 */
void 
Pooling::issueLayer(ThreadArg* threadArg)
{
    MMU* mmu = threadArg->mmu;

    pthread_mutex_lock ( ioMutex );
        vector<unsigned long long> iFMapPages   = mmu->addressTranslate(iFMap.first);
        vector<unsigned long long> oFMapPages   = mmu->addressTranslate(oFMap.first);
        vector<unsigned long long> filterPages  = mmu->addressTranslate(filter.first);
        log_V("iFMapPages Num"  , to_string(iFMapPages.size()));
        log_V("oFMapPages Num"  , to_string(oFMapPages.size()));
        log_V("filterPages Num" , to_string(filterPages.size()));
    pthread_mutex_unlock ( ioMutex );

    int data_byte = sizeof(DATA_TYPE);
    
    /* Thread compile start and end index */
    int start_index = (oFMapSize[WIDTH] * threadArg->threadID) / threadArg->numThread;
    int end_index   = (oFMapSize[WIDTH] * (threadArg->threadID + 1)) / threadArg->numThread;

    /* Use inverse order for let the address be closer */
    int index = 0;
    for (int w_o = start_index; w_o < end_index; w_o++)
    {
        for (int h_o = 0; h_o < oFMapSize[HEIGHT]; h_o++)
        {
            for (int c_o = 0; c_o < oFMapSize[CHANNEL]; c_o++)
            {
                for (int b = 0; b < oFMapSize[BATCH]; b++)
                {                   
                    Request* request = new Request();
                    
                    const int h_start = h_o * stride[STRIDE_PADDING_HEIGHT] - padding[STRIDE_PADDING_HEIGHT];
                    const int w_start = w_o * stride[STRIDE_PADDING_HEIGHT] - padding[STRIDE_PADDING_HEIGHT];

                    for (int c_i = 0; c_i < filterSize[FILTER_CHANNEL_I]; c_i++)
                    {
                        /* read filter pages */
                        for (int h_i = max(0, h_start); h_i < min(h_start + filterSize[HEIGHT], iFMapSize[HEIGHT]); h_i++)
                        {
                            for (int w_i = max(0, w_start); w_i < min(w_start + filterSize[WIDTH], iFMapSize[WIDTH]); w_i++)
                            {
                                /* read filter pages */
                                index = floor((c_i * filterSize[HEIGHT] * filterSize[WIDTH] + h_i * filterSize[WIDTH] + w_i) * data_byte / PAGE_SIZE);
                                ASSERT(index < filterPages.size(), "Layer " + to_string(layerID) + " (" + layerType + ") Overflow!");

                                request->readPages.emplace_back(make_pair(filterPages[index], 1));
                                
                                /* read input pages */
                                index = floor((b * iFMapSize[CHANNEL] * iFMapSize[HEIGHT] * iFMapSize[WIDTH] + c_i * iFMapSize[HEIGHT] * iFMapSize[WIDTH] + h_i * iFMapSize[WIDTH] + w_i) * data_byte / PAGE_SIZE);
                                ASSERT(index < iFMapPages.size(), "Layer " + to_string(layerID) + " (" + layerType + ") Overflow!");

                                request->readPages.emplace_back(make_pair(iFMapPages[index], 1));
                            }                       
                        }
                    }

                    // Pooling layer find the maxinum input data in the field masked by filter
                    request->numOfInstructions = filterSize[HEIGHT] * filterSize[WIDTH];

                    /* write result to pages */
                    index = floor((b * oFMapSize[CHANNEL] * oFMapSize[HEIGHT] * oFMapSize[WIDTH] + c_o * oFMapSize[HEIGHT] * oFMapSize[WIDTH] + h_o * oFMapSize[WIDTH] + w_o) * data_byte / PAGE_SIZE);
                    ASSERT(index < oFMapPages.size(), "Layer " + to_string(layerID) + " (" + layerType + ") Overflow!");

                    request->writePages.emplace_back(make_pair(oFMapPages[index], data_byte));

                    /* for the activation exectuion */
                    if (strcmp(activationType, "None") != 0) request->numOfInstructions++;  // for the activation exectuion

                    threadArg->requestQueue->push(move(Kernel::compressRequest(request)));
                }
                
            }
            
        }
        
    }

    delete threadArg;
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
Flatten::Flatten(int layer_id, vector<int> input_size) : Layer(layer_id, (char*)"Flatten", input_size)
{
    calculateOFMapSize();
    int size = oFMapSize[BATCH] * oFMapSize[CHANNEL] * oFMapSize[HEIGHT] * oFMapSize[WIDTH];
    oFMap = make_pair(++vaCount, new vector<DATA_TYPE>(size));
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
    ASSERT(!iFMapSize.empty(), "Cannot calculate the size of OFMap due to missing parameter.");

    oFMapSize.emplace_back(iFMapSize[BATCH]);
    oFMapSize.emplace_back(iFMapSize[CHANNEL] * iFMapSize[HEIGHT] * iFMapSize[WIDTH]);
    oFMapSize.emplace_back(1);
    oFMapSize.emplace_back(1);
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

    std::cout << std::left << std::setw(10) << layerID
              << std::left << std::setw(16) << layerType 
              << std::left << std::setw(13) << activationType;

    std::cout << "(" 
              << std::right << std::setw(4)  << iFMapSize[BATCH]   << ", "
              << std::right << std::setw(4)  << iFMapSize[CHANNEL] << ", "
              << std::right << std::setw(3)  << iFMapSize[HEIGHT]  << ", "
              << std::right << std::setw(3)  << iFMapSize[WIDTH]   
              << std::left  << std::setw(10) << ")"         
              << std::right << std::setw(22) << "None"  
              << std::right << setw(10) << "("
              << std::right << std::setw(4)  << oFMapSize[BATCH]   << ", "
              << std::right << std::setw(4)  << oFMapSize[CHANNEL] << ", " 
              << std::right << std::setw(3)  << oFMapSize[HEIGHT]  << ", " 
              << std::right << std::setw(3)  << oFMapSize[WIDTH]   
              << std::left  << std::setw(10) << ")";  

    std::cout << std::endl;
}


/** ===============================================================================================
 * \name    issueLayer
 *
 * \brief   Flatten layer casting the input dimension into a 1-dim array.
 * 
 * \note    The batch size must be designed
 * 
 * \note    The memory address must be allocated
 * 
 * \param   arg     the void* of threadArg
 * 
 * \endcond
 * ================================================================================================
 */
void 
Flatten::issueLayer(ThreadArg* threadArg)
{
    MMU* mmu = threadArg->mmu;

    pthread_mutex_lock ( ioMutex );
        vector<unsigned long long> iFMapPages   = mmu->addressTranslate(iFMap.first);
        vector<unsigned long long> oFMapPages   = mmu->addressTranslate(oFMap.first);
        log_V("iFMapPages Num"  , to_string(iFMapPages.size()));
        log_V("oFMapPages Num"  , to_string(oFMapPages.size()));
    pthread_mutex_unlock ( ioMutex );

    int data_byte = sizeof(DATA_TYPE);

    /* Thread compile start and end index */
    int start_index = (oFMapSize[WIDTH] * threadArg->threadID) / threadArg->numThread;
    int end_index   = (oFMapSize[WIDTH] * (threadArg->threadID + 1)) / threadArg->numThread;

    int index = 0;
    for (int w_o = start_index; w_o < end_index; w_o++)
    {
        for (int h_o = 0; h_o < oFMapSize[HEIGHT]; h_o++)
        {
            for (int c_o = 0; c_o < oFMapSize[CHANNEL]; c_o++)
            {
                for (int b = 0; b < oFMapSize[BATCH]; b++)
                {                   
                    Request* request = new Request();
                    
                    /* read input pages && write result to pages */
                    index = floor((b * oFMapSize[CHANNEL] * oFMapSize[HEIGHT] * oFMapSize[WIDTH] + c_o * oFMapSize[HEIGHT] * oFMapSize[WIDTH] + h_o * oFMapSize[WIDTH] + w_o) * data_byte / PAGE_SIZE);
                    ASSERT(index < iFMapPages.size() && index < oFMapPages.size(), "Layer " + to_string(layerID) + " (" + layerType + ") Overflow!");
                    
                    request->readPages.emplace_back(make_pair(iFMapPages[index], 1));
                    request->writePages.emplace_back(make_pair(oFMapPages[index], 1));

                    threadArg->requestQueue->push(move(Kernel::compressRequest(request)));
                }
            }  
        }
    }

    delete threadArg;
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
ByPass::ByPass(int layer_id, vector<int> input_size) : Layer(layer_id, (char*)"ByPass", input_size)
{
    
    ASSERT(!iFMapSize.empty(), "Cannot calculate the size of OFMap due to missing parameter.");
    oFMapSize = iFMapSize;
    oFMap = make_pair(++vaCount, new vector<DATA_TYPE>(oFMapSize[BATCH] * oFMapSize[CHANNEL] * oFMapSize[HEIGHT] * oFMapSize[WIDTH]));
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
ByPass::ByPass(int layer_id, vector<int> input_size, vector<int> output_size)
        : Layer(layer_id, (char*)"ByPass", input_size)
{
    ASSERT(!output_size.empty(), "Cannot calculate the size of OFMap due to missing parameter.");
    oFMapSize = output_size;
    oFMap = make_pair(++vaCount, new vector<DATA_TYPE>(output_size[BATCH] * output_size[CHANNEL] * output_size[HEIGHT] * output_size[WIDTH]));
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

    std::cout << std::left << std::setw(10) << layerID
              << std::left << std::setw(16) << layerType 
              << std::left << std::setw(13) << activationType;

    std::cout << "(" 
              << std::right << std::setw(4)  << iFMapSize[BATCH]   << ", " 
              << std::right << std::setw(4)  << iFMapSize[CHANNEL] << ", " 
              << std::right << std::setw(3)  << iFMapSize[HEIGHT]  << ", " 
              << std::right << std::setw(3)  << iFMapSize[WIDTH]  
              << std::left  << std::setw(10) << ")"  
              << std::right << std::setw(22) << "None"                        
              << std::right << setw(10) << "("
              << std::right << std::setw(4)  << oFMapSize[BATCH]   << ", " 
              << std::right << std::setw(4)  << oFMapSize[CHANNEL] << ", " 
              << std::right << std::setw(3)  << oFMapSize[HEIGHT]  << ", " 
              << std::right << std::setw(3)  << oFMapSize[WIDTH]  
              << std::left  << std::setw(10) << ")";  

    std::cout << std::endl;
}


/** ===============================================================================================
 * \name    issueLayer
 *
 * \brief   ByPass layer is my self designed layer for tranfer the previous result.
 * 
 * \note    The batch size must be designed
 * 
 * \note    The memory address must be allocated
 * 
 * \param   arg     the void* of threadArg
 * 
 * \note    After added GoogleNet, ByPass layer also used for the I/O dimension mapping. Therefore,
 *          cannot use the oFMapSize to compile the requests which intreduces segmentation fault.
 * 
 * \endcond
 * ================================================================================================
 */
void 
ByPass::issueLayer(ThreadArg* threadArg)
{
    MMU* mmu = threadArg->mmu;
    
    pthread_mutex_lock ( ioMutex );
        vector<unsigned long long> iFMapPages   = mmu->addressTranslate(iFMap.first);
        vector<unsigned long long> oFMapPages   = mmu->addressTranslate(oFMap.first);
        log_V("iFMapPages Num"  , to_string(iFMapPages.size()));
        log_V("oFMapPages Num"  , to_string(oFMapPages.size()));
    pthread_mutex_unlock ( ioMutex );

    int data_byte = sizeof(DATA_TYPE);

    /* Thread compile start and end index */
    int start_index = (iFMapSize[WIDTH] * threadArg->threadID) / threadArg->numThread;
    int end_index   = (iFMapSize[WIDTH] * (threadArg->threadID + 1)) / threadArg->numThread;

    int index = 0;
    for (int w_i = start_index; w_i < end_index; w_i++)
    {
        for (int h_i = 0; h_i < iFMapSize[HEIGHT]; h_i++)
        {
            for (int c_i = 0; c_i < iFMapSize[CHANNEL]; c_i++)
            {
                for (int b = 0; b < iFMapSize[BATCH]; b++)
                {                   
                    Request* request = new Request();
                    
                    /* read input pages && write result to pages */
                    index = floor((b * iFMapSize[CHANNEL] * iFMapSize[HEIGHT] * iFMapSize[WIDTH] + c_i * iFMapSize[HEIGHT] * iFMapSize[WIDTH] + h_i * iFMapSize[WIDTH] + w_i) * data_byte / PAGE_SIZE);
                    ASSERT(index < iFMapPages.size() && index < oFMapPages.size(), "Layer " + to_string(layerID) + " (" + layerType + ") Overflow!");
                    
                    request->readPages.emplace_back(make_pair(iFMapPages[index], 1));
                    request->writePages.emplace_back(make_pair(oFMapPages[index], 1));

                    threadArg->requestQueue->push(move(Kernel::compressRequest(request)));
                }
            }  
        }
    }
    
    delete threadArg;
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
Dense::Dense(int layer_id, vector<int> input_size, vector<int> filter_size, char* activation_type)
        : Layer(layer_id, (char*)"Dense", input_size, filter_size, activation_type)
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
Dense::Dense(int layer_id, vector<int> input_size, int output_width)
        : Dense(layer_id, input_size, {output_width, input_size[CHANNEL], 1, 1}, (char*)"Relu")
{
    calculateOFMapSize();
    int size = oFMapSize[BATCH] * oFMapSize[CHANNEL] * oFMapSize[HEIGHT] * oFMapSize[WIDTH];
    oFMap = make_pair(++vaCount, new vector<DATA_TYPE>(size));
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
    ASSERT(!iFMapSize.empty(), "Cannot calculate the size of OFMap due to missing parameter.");

    oFMapSize.emplace_back(iFMapSize[BATCH]);
    oFMapSize.emplace_back(filterSize[FILTER_CHANNEL_O]);
    oFMapSize.emplace_back(1);
    oFMapSize.emplace_back(1);
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

    std::cout << std::left << std::setw(10) << layerID 
              << std::left << std::setw(16) << layerType 
              << std::left << std::setw(13) << activationType;

    std::cout << "(" 
              << std::right << std::setw(4)  << iFMapSize[BATCH]   << ", " 
              << std::right << std::setw(4)  << iFMapSize[CHANNEL] << ", " 
              << std::right << std::setw(3)  << iFMapSize[HEIGHT]  << ", " 
              << std::right << std::setw(3)  << iFMapSize[WIDTH]  
              << std::left  << std::setw(10) << ")"
              << std::right << std::setw(22) << "None"   
              << std::right << setw(10) << "("  
              << std::right << std::setw(4)  << oFMapSize[BATCH]   << ", " 
              << std::right << std::setw(4)  << oFMapSize[CHANNEL] << ", " 
              << std::right << std::setw(3)  << oFMapSize[HEIGHT]  << ", " 
              << std::right << std::setw(3)  << oFMapSize[WIDTH] 
              << std::left  << std::setw(10) << ")"; 

    std::cout << std::endl;
}


/** ===============================================================================================
 * \name    issueLayer
 *
 * \brief   Dense layer use linear transormation function to down / up sampling the data dimension.
 * 
 * \note    The batch size must be designed
 * 
 * \note    The memory address must be allocated
 * 
 * \param   arg     the void* of threadArg
 * 
 * \endcond
 * ================================================================================================
 */
void 
Dense::issueLayer(ThreadArg* threadArg)
{
    MMU* mmu = threadArg->mmu;

    pthread_mutex_lock ( ioMutex );
        vector<unsigned long long> iFMapPages   = mmu->addressTranslate(iFMap.first);
        vector<unsigned long long> oFMapPages   = mmu->addressTranslate(oFMap.first);
        vector<unsigned long long> filterPages  = mmu->addressTranslate(filter.first);
        log_V("iFMapPages Num"  , to_string(iFMapPages.size()));
        log_V("oFMapPages Num"  , to_string(oFMapPages.size()));
        log_V("filterPages Num" , to_string(filterPages.size()));
    pthread_mutex_unlock ( ioMutex );

    ASSERT(iFMapSize[HEIGHT] == 1 && iFMapSize[WIDTH] == 1, "Dimension error!");

    int data_byte = sizeof(DATA_TYPE);

    /* Thread compile start and end index */
    int start_index = (oFMapSize[CHANNEL] * threadArg->threadID) / threadArg->numThread;
    int end_index   = (oFMapSize[CHANNEL] * (threadArg->threadID + 1)) / threadArg->numThread;

    /* Use inverse order for let the address be closer */
    int index = 0;
    for (int c_o = start_index; c_o < end_index; c_o++)
    {
        for (int b = 0; b < oFMapSize[BATCH]; b++)
        {                   
            Request* request = new Request();

            for (int c_i = 0; c_i < iFMapSize[CHANNEL]; c_i++)
            {
                /* read filter pages */
                index = floor((c_o * filterSize[FILTER_CHANNEL_I] + c_i) * data_byte / PAGE_SIZE);
                ASSERT(index < filterPages.size(), "Layer " + to_string(layerID) + " (" + layerType + ") Overflow!");

                request->readPages.emplace_back(make_pair(filterPages[index], 1));

                /* read input pages */
                index = floor((b * iFMapSize[CHANNEL] + c_i) * data_byte / PAGE_SIZE);
                ASSERT(index < iFMapPages.size(),  "Layer " + to_string(layerID) + " (" + layerType + ") Overflow!");

                request->readPages.emplace_back(make_pair(iFMapPages[index], 1));
            }

            // Performs dot product
            request->numOfInstructions = filterSize[FILTER_CHANNEL_O] * filterSize[FILTER_CHANNEL_I];

            /* write result to pages */
            index = floor((b * oFMapSize[CHANNEL] + c_o) * data_byte / PAGE_SIZE);
            ASSERT(index < oFMapPages.size(), "Layer " + to_string(layerID) + " (" + layerType + ") Overflow!");

            request->writePages.emplace_back(make_pair(oFMapPages[index], 1));

            threadArg->requestQueue->push(move(Kernel::compressRequest(request)));

        }
    }
    
    delete threadArg;
}

