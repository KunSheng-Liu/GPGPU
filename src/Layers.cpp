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
 * \param   output_size         [batch, channel, height, width]
 * \param   filter_size         [channel, height, width]
 * \param   activation_type     the activation type
 * 
 * \endcond
 * ================================================================================================
 */
Layer::Layer(char* layer_type, int* input_size, int* output_size, int* filter_size, char* activation_type)
        : layerType(layer_type), iFMapSize(input_size), oFMapSize(output_size), filterSize(filter_size), activationType(activation_type)
        , layerIndex(++layerCount), flagExecuting(false), flagFinish(false)
{
    /* Seems no need to initial the data, due to it come from previous layer */
    iFMap  = (iFMapSize  == NULL)  ? NULL : new unsigned char[iFMapSize[BATCH] * iFMapSize[CHANNEL] * iFMapSize[HEIGHT] * iFMapSize[WIDTH]];
    oFMap  = (oFMapSize  == NULL)  ? NULL : new unsigned char[oFMapSize[BATCH] * oFMapSize[CHANNEL] * oFMapSize[HEIGHT] * oFMapSize[WIDTH]];
    filter = (filterSize  == NULL) ? NULL : new unsigned char[filterSize[FILTER_CHANNEL_I] * filterSize[FILTER_CHANNEL_O] * filterSize[HEIGHT] * filterSize[WIDTH]];
    
}


/** ===============================================================================================
 * \name   ~Layer
 *
 * \brief   Destruct a layer
 * 
 * \endcond
 * ================================================================================================
 */
Layer::Layer::~Layer()
{
    if (strcmp (layerType, "None") != 0)
    {
        (iFMap  == NULL) ? void(0) : delete [] iFMap;
        (oFMap  == NULL) ? void(0) : delete [] oFMap;
        (filter == NULL) ? void(0) : delete [] filter;

        (iFMapSize  == NULL) ? void(0) : delete [] iFMapSize;
        (oFMapSize  == NULL) ? void(0) : delete [] oFMapSize;
        (filterSize == NULL) ? void(0) : delete [] filterSize;
    }
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
              << std::right << std::setw(3)  << filterSize[FILTER_CHANNEL_I] << ", " \
              << std::right << std::setw(3)  << filterSize[FILTER_CHANNEL_O] << ", " \
              << std::right << std::setw(4)  << filterSize[HEIGHT]           << ", " \
              << std::right << std::setw(4)  << filterSize[WIDTH]                    \
              << std::left  << std::setw(10) << ")" << "("                           \
              << std::right << std::setw(3)  << oFMapSize[BATCH]             << ", " \
              << std::right << std::setw(3)  << oFMapSize[CHANNEL]           << ", " \
              << std::right << std::setw(4)  << oFMapSize[HEIGHT]            << ", " \
              << std::right << std::setw(3)  << oFMapSize[WIDTH]                     \
              << std::left  << std::setw(10) << ")"; 
}



/** ===============================================================================================
 * \name    Conv2D
 *
 * \brief   For the inheritance class
 * 
 * \param   layer_type          the layer type
 * \param   input_size          [batch, channel, height, width]
 * \param   filter_size         [channel, height, width]
 * \param   activation_type     the activation type
 * \param   _stride             [height, width]
 * \param   _padding            [height, width]
 * 
 * \endcond
 * ================================================================================================
 */
Conv2D::Conv2D(char* layer_type, int* input_size, int* filter_size, char* activation_type, int* _stride, int* _padding)
        : Layer(layer_type, move(input_size), NULL, move(filter_size), activation_type)
        , stride(_stride), padding(_padding)
{
    ASSERT(iFMapSize != NULL, "error input");
    oFMapSize = calculateOFMapSize();
    oFMap  = (oFMapSize  == NULL)  ? NULL : new unsigned char[oFMapSize[BATCH] * oFMapSize[CHANNEL] * oFMapSize[HEIGHT] * oFMapSize[WIDTH]];
}


/** ===============================================================================================
 * \name    Conv2D
 *
 * \brief   Construct a 2D convolution layer
 * 
 * \param   input_size          [batch, channel, height, width]
 * \param   filter_size         [channel, height, width]
 * \param   activation_type     the activation type
 * \param   _stride             [height, width]
 * \param   _padding            [height, width]
 * 
 * \endcond
 * ================================================================================================
 */
Conv2D::Conv2D(int* input_size, int* filter_size, char* activation_type, int* _stride, int* _padding)
        : Conv2D((char*)"Conv2D"
        , move(input_size), move(filter_size)
        , activation_type, move(_stride), move(_padding))
{

}


/** ===============================================================================================
 * \name    Conv2D
 *
 * \brief   Construct a 2D convolution layer
 * 
 * \param   input_size          [batch, channel, height, width]
 * \param   filter_size         [channel, height, width]
 * \param   activation_type     the activation type
 * \param   _stride             width
 * \param   _padding            width
 * 
 * \endcond
 * ================================================================================================
 */
Conv2D::Conv2D(int* input_size, int* filter_size, char* activation_type, int _stride, int _padding)
        : Conv2D((char*)"Conv2D"
        , move(input_size), move(filter_size), activation_type
        , move(new int[2]{_stride,_stride}), move(new int[2]{_padding, _padding}))
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
    (stride  == NULL) ? void(0) : delete [] stride;
    (padding  == NULL) ? void(0) : delete [] padding;
}


/** ===============================================================================================
 * \name    Conv2D
 *
 * \brief   Calculate the output feature map dimension
 * 
 * \endcond
 * ================================================================================================
 */
int*
Conv2D::calculateOFMapSize()
{
    bool check = (iFMapSize != NULL) && (filterSize != NULL) && (stride != NULL) && (padding != NULL);
    ASSERT(check, "Cannot calculate the size of OFMap due to missing parameter.");

    int* shape = new int[4];
    shape[BATCH]   = iFMapSize[BATCH];
    shape[CHANNEL] = filterSize[FILTER_CHANNEL_O];
    shape[HEIGHT]  = (double) (iFMapSize[HEIGHT] + 2 * padding[STRIDE_PADDING_HEIGHT] - filterSize[HEIGHT]) / (double) (stride[STRIDE_PADDING_HEIGHT]) + 1;
    shape[WIDTH]   = (double) (iFMapSize[WIDTH]  + 2 * padding[STRIDE_PADDING_WIDTH]  - filterSize[WIDTH])  / (double) (stride[STRIDE_PADDING_WIDTH])  + 1;
    
    return move(shape);
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
Conv2D::printInfo()
{
    std::cout << std::left << std::setw(10) << layerIndex; 
    std::cout << std::left << std::setw(10) << layerType; 
    std::cout << std::left << std::setw(10) << activationType;
    Layer::printInfo();
    std::cout << "(" << std::right << std::setw(2)  << stride[STRIDE_PADDING_HEIGHT]  << ", " \
                     << std::right << std::setw(2)  << stride[STRIDE_PADDING_WIDTH]           \
                     << std::left  << std::setw(10) << ")" << "("                             \
                     << std::right << std::setw(2)  << padding[STRIDE_PADDING_HEIGHT] << ", " \
                     << std::right << std::setw(2)  << padding[STRIDE_PADDING_WIDTH]  << ")" << std::endl;
}


/** ===============================================================================================
 * \name    issueLayer
 *
 * \brief   Print the layer information
 * 
 * \endcond
 * ================================================================================================
 */
void
Conv2D::issueLayer()
{

}



/** ===============================================================================================
 * \name    Pooling
 *
 * \brief   Construct a pooling layer
 * 
 * \param   input_size          [batch, channel, height, width]
 * \param   filter_size         [channel, height, width]
 * \param   activation_type     the activation type
 * \param   _stride             [height, width]
 * \param   _padding            [height, width]
 * 
 * \endcond
 * ================================================================================================
 */
Pooling::Pooling(int* input_size, int* filter_size, char* activation_type, int* _stride, int* _padding)
        : Conv2D((char*)"Pooling", move(input_size), move(filter_size), activation_type, move(_stride), move(_padding))
{
    
}


/** ===============================================================================================
 * \name    Pooling
 *
 * \brief   Construct a pooling layer
 * 
 * \param   input_size          [batch, channel, height, width]
 * \param   filter_size         [channel, height, width]
 * \param   activation_type     the activation type
 * \param   _stride             [height, width]
 * \param   _padding            [height, width]
 * 
 * \endcond
 * ================================================================================================
 */
Pooling::Pooling(int* input_size, int* filter_size, char* activation_type, int _stride, int _padding)
        : Conv2D((char*)"Pooling"
        , move(input_size), move(filter_size), activation_type
        , move(new int[2]{_stride,_stride}), move(new int[2]{_padding, _padding}))
{
    
}


/** ===============================================================================================
 * \name    issueLayer
 *
 * \brief   Print the layer information
 * 
 * \endcond
 * ================================================================================================
 */
void
Pooling::issueLayer()
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
Flatten::Flatten(int* input_size)
        : Layer((char*)"Flatten", move(input_size), NULL, NULL, (char*)"None")
{
    oFMapSize = calculateOFMapSize();
    oFMap  = (oFMapSize  == NULL) ? NULL : new unsigned char[oFMapSize[BATCH] * oFMapSize[CHANNEL] * oFMapSize[HEIGHT] * oFMapSize[WIDTH]];
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
int*
Flatten::calculateOFMapSize()
{
    bool check = (iFMapSize != NULL);
    ASSERT(check, "Cannot calculate the size of OFMap due to missing parameter.");

    int* shape = new int[4];
    shape[BATCH]   = iFMapSize[BATCH];
    shape[CHANNEL] = iFMapSize[CHANNEL] * iFMapSize[HEIGHT] * iFMapSize[WIDTH];
    shape[HEIGHT]  = 1;
    shape[WIDTH]   = 1;

    return move(shape);
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
              << std::left << std::setw(10) << layerType  \
              << std::left << std::setw(10) << activationType;

    std::cout << std::right << std::setw(22) << "None"                \
              << std::right << setw(10) << "("
              << std::right << std::setw(3)  << oFMapSize[BATCH]   << ", " \
              << std::right << std::setw(5)  << oFMapSize[CHANNEL] << ", " \
              << std::right << std::setw(2)  << oFMapSize[HEIGHT]  << ", " \
              << std::right << std::setw(3)  << oFMapSize[WIDTH]           \
              << std::left  << std::setw(10) << ")";  

    std::cout << std::endl;
}


/** ===============================================================================================
 * \name    issueLayer
 *
 * \brief   Print the layer information
 * 
 * \endcond
 * ================================================================================================
 */
void
Flatten::issueLayer()
{

}



/** ===============================================================================================
 * \name    Dense
 *
 * \brief   Construct a dense layer
 * 
 * \param   input_size          [batch, channel, height, width]
 * \param   filter_size         [channel, height, width]
 * \param   activation_type     the activation type
 * 
 * \endcond
 * ================================================================================================
 */
Dense::Dense(int* input_size, int* filter_size, char* activation_type)
        : Layer((char*)"Dense", move(input_size), NULL, move(filter_size), activation_type)
{
    oFMapSize = calculateOFMapSize();
    oFMap  = (oFMapSize  == NULL) ? NULL : new unsigned char[oFMapSize[BATCH] * oFMapSize[CHANNEL] * oFMapSize[HEIGHT] * oFMapSize[WIDTH]];
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
Dense::Dense(int* input_size, int output_width)
        : Layer((char*)"Dense", move(input_size), NULL, new int[4]{input_size[CHANNEL], output_width, 1, 1})
{
    oFMapSize = calculateOFMapSize();
    oFMap  = (oFMapSize  == NULL) ? NULL : new unsigned char[oFMapSize[BATCH] * oFMapSize[CHANNEL] * oFMapSize[HEIGHT] * oFMapSize[WIDTH]];
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
int*
Dense::calculateOFMapSize()
{
    bool check = (iFMapSize != NULL) && (filterSize != NULL);
    ASSERT(check, "Cannot calculate the size of OFMap due to missing parameter.");

    int* shape = new int[4];
    shape[BATCH]   = iFMapSize[BATCH];
    shape[CHANNEL] = filterSize[FILTER_CHANNEL_O];
    shape[HEIGHT]  = 1;
    shape[WIDTH]   = 1;

    return move(shape);
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
              << std::left << std::setw(10) << layerType  \
              << std::left << std::setw(10) << activationType;

    std::cout << "("                        
              << std::right << std::setw(3)  << iFMapSize[BATCH] << ", " \
              << std::right << std::setw(5)  << iFMapSize[CHANNEL] << ", " \
              << std::right << std::setw(2)  << iFMapSize[HEIGHT]           << ", " \
              << std::right << std::setw(4)  << iFMapSize[WIDTH]                    \
              << std::left  << std::setw(10) << ")" << "("                           \
              << std::right << std::setw(3)  << oFMapSize[BATCH]             << ", " \
              << std::right << std::setw(5)  << oFMapSize[CHANNEL]           << ", " \
              << std::right << std::setw(2)  << oFMapSize[HEIGHT]            << ", " \
              << std::right << std::setw(3)  << oFMapSize[WIDTH]                     \
              << std::left  << std::setw(10) << ")"; 

    std::cout << std::endl;
}


/** ===============================================================================================
 * \name    issueLayer
 *
 * \brief   Print the layer information
 * 
 * \endcond
 * ================================================================================================
 */
void
Dense::issueLayer()
{

}

