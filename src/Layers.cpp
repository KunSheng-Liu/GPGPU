/**
 * \name    Layer.cpp
 * 
 * \brief   Implement the API
 * 
 * \date    Mar 31, 2023
 */

#include "include/Layers.hpp"

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
Layer::Layer(Layer_t layer_type, int* input_size, int* output_size, int* filter_size, Activation_t activation_type)
        : layerType(layer_type), iFMapSize(input_size), oFMapSize(output_size), filterSize(filter_size), activationType(activation_type)
        , layerIndex(-1), flagExecuting(false), flagFinish(false)
{
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
    (iFMap  == NULL) ? void(0) : delete [] iFMap;
    (oFMap  == NULL) ? void(0) : delete [] oFMap;
    (filter == NULL) ? void(0) : delete [] filter;

    delete [] iFMapSize;
    delete [] filterSize;
    delete [] oFMapSize;
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
Conv2D::Conv2D(int* input_size, int* filter_size, Activation_t activation_type, int* _stride, int* _padding)
        : Layer(Layer_t::Conv2D, move(input_size), NULL, move(filter_size), activation_type)
        , stride(_stride), padding(_padding)
{
    oFMapSize = calculateOFMapSize();
    oFMap  = (oFMapSize  == NULL)  ? NULL : new unsigned char[oFMapSize[BATCH] * oFMapSize[CHANNEL] * oFMapSize[HEIGHT] * oFMapSize[WIDTH]];
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
Conv2D::Conv2D(Layer_t layer_type, int* input_size, int* filter_size, Activation_t activation_type, int* _stride, int* _padding)
        : Layer(layer_type, move(input_size), NULL, move(filter_size), activation_type)
        , stride(_stride), padding(_padding)
{
    oFMapSize = calculateOFMapSize();
    oFMap  = (oFMapSize  == NULL)  ? NULL : new unsigned char[oFMapSize[BATCH] * oFMapSize[CHANNEL] * oFMapSize[HEIGHT] * oFMapSize[WIDTH]];
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
    shape[HEIGHT]  = (double) (iFMapSize[HEIGHT] + 2 * padding[HEIGHT] - filterSize[HEIGHT]) / (double) (stride[HEIGHT]) + 1;
    shape[WIDTH]   = (double) (iFMapSize[WIDTH]  + 2 * padding[WIDTH]  - filterSize[WIDTH])  / (double) (stride[WIDTH])  + 1;

    return move(shape);
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
Pooling::Pooling(int* input_size, int* filter_size, Activation_t activation_type, int* _stride, int* _padding)
        : Conv2D(Layer_t::Pooling, move(input_size), move(filter_size), activation_type, move(_stride), move(_padding))
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
Dense::Dense(int* input_size, int* filter_size, Activation_t activation_type)
        : Layer(Layer_t::Conv2D, move(input_size), NULL, move(filter_size), activation_type)
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



