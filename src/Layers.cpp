/**
 * \name    Layer.cpp
 * 
 * \brief   Implement the API
 * 
 * \date    Mar 31, 2023
 */

#include "include/Layers.h"

/** ===============================================================================================
 * \name    Layer
 *
 * \brief   Construct a layer
 * 
 * \param   prev_layer          the pointer of previous layer
 * \param   layer_type          the layer type
 * \param   input_size          [batch, channel, width, height]
 * \param   filter_size         [channel, width, height]
 * \param   activation_type     the activation type
 * 
 * \endcond
 * ================================================================================================
 */
Layer::Layer(Layer* prev_layer, Layer_t layer_type, int* input_size, int* filter_size, Activation_t activation_type)
        : layerIndex(-1), layerType(layer_type), activationType(activation_type)
{
    /* Set up input feature map size */
    iFMapSize = new int[4];
    iFMapSize[BATCH]    = input_size[0];
    iFMapSize[CHANNEL]  = input_size[1];
    iFMapSize[HEIGHT]   = input_size[2];
    iFMapSize[WIDTH]    = input_size[3];

    /* Set up filter size */
    int* filterSize = new int[4];
    filterSize[FILTER_CHANNEL_I] = filter_size[0];
    filterSize[FILTER_CHANNEL_O] = filter_size[1];
    filterSize[HEIGHT] = filter_size[2];
    filterSize[WIDTH]  = filter_size[3];

    /* Set up connection */
    setUpConnection (prev_layer);
}


/** ===============================================================================================
 * \name    Layer
 *
 * \brief   Construct a layer
 * 
 * \param   prev_layer          the pointer of previous layer
 * \param   layer_type          the layer type
 * \param   input_size          input_size = [1, input_channel, input_width, input_width]
 * \param   input_channel       input channel
 * \param   output_channel      output channel
 * \param   filter_width        the kernel width
 * \param   activation_type     the activation type
 * 
 * \endcond
 * ================================================================================================
 */
Layer::Layer(Layer* prev_layer, Layer_t layer_type, int* input_size, int input_channel,
        int output_channel, int filter_width, Activation_t activation_type)
{
    int* filter_size = new int[4];
    filter_size[FILTER_CHANNEL_I] = input_channel;
    filter_size[FILTER_CHANNEL_O] = output_channel;
    filter_size[HEIGHT] = filter_width;
    filter_size[WIDTH]  = filter_width;

    Layer(prev_layer, layer_type, input_size, filter_size, activation_type);
}


/** ===============================================================================================
 * \name    Layer
 *
 * \brief   Construct a layer
 * 
 * \param   prev_layer          the pointer of previous layer
 * \param   layer_type          the layer type
 * \param   input_width         input width
 * \param   input_channel       input channel
 * \param   output_channel      output channel
 * \param   filter_width        the kernel width
 * \param   activation_type     the activation type
 * 
 * \endcond
 * ================================================================================================
 */
Layer::Layer(Layer* prev_layer, Layer_t layer_type, int input_width, int input_channel,
        int output_channel, int filter_width, Activation_t activation_type)
{
    int* input_size = new int[4];
    input_size[BATCH]   = 1;
    input_size[CHANNEL] = input_channel;
    input_size[HEIGHT]  = input_width;
    input_size[WIDTH]   = input_width;

    int* filter_size = new int[4];
    filter_size[FILTER_CHANNEL_I] = input_channel;
    filter_size[FILTER_CHANNEL_O] = output_channel;
    filter_size[HEIGHT] = filter_width;
    filter_size[WIDTH]  = filter_width;

    Layer(prev_layer, layer_type, input_size, filter_size, activation_type);
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
 * \name    init
 *
 * \brief   Initial the oFMapSize, defualt stride is "1", padding is "0"
 * 
 * \endcond
 * 
 * The output feature map is calculated by the formula: floor{(iFMap - Kernel + 2 * padding) / stride} + 1
 * 
 * ================================================================================================
 */
void
Layer::init()
{
    oFMapSize = new int[4];
    oFMapSize[BATCH]    = iFMapSize[BATCH];
    oFMapSize[CHANNEL]  = filterSize[FILTER_CHANNEL_O];
    oFMapSize[HEIGHT]   = (double) (iFMapSize[HEIGHT] - filterSize[HEIGHT]) / (double) (1) + 1;
    oFMapSize[WIDTH]    = (double) (iFMapSize[WIDTH] - filterSize[WIDTH]) / (double) (1) + 1;
}


/** ===============================================================================================
 * \name    setUpConnection
 *
 * \brief   Set up the connection between current layer and previou layer.
 * 
 * \param   prev_layer pointer of previou layer
 * 
 * \endcond
 * ================================================================================================
 */
void
Layer::setUpConnection(Layer* prev_layer)
{
    ASSERT( prev_layer != this, "Try to set current layer as previous layer" );

    if (prev_layer != NULL) {

        addPrevLayer(prev_layer);
        prev_layer->addNextLayer(this);

    }
}


/** ===============================================================================================
 * \name    addPrevLayer
 *
 * \brief   Add the next layer pointer into \b "nextLayer"
 * 
 * \param   prev_layer pointer of previou layer
 * 
 * \endcond
 * ================================================================================================
 */
void 
Layer::addPrevLayer(Layer* prev_layer)
{
    ASSERT( prev_layer, "NULL previous layer" );

    prevLayer.emplace_back(prev_layer);
}


/** ===============================================================================================
 * \name    addNextLayer
 *
 * \brief   Add the next layer pointer into \b "nextLayer"
 * 
 * \param   prev_layer pointer of previou layer
 * 
 * \endcond
 * ================================================================================================
 */
void 
Layer::addNextLayer(Layer* next_layer)
{
    ASSERT( next_layer, "NULL next layer" );

    nextLayer.emplace_back(next_layer);
}


/** ===============================================================================================
 * \name    isFinish
 *
 * \brief   Check the 
 * 
 * \param   prev_layer pointer of previou layer
 * 
 * \endcond
 * ================================================================================================
 */
void 
Layer::addNextLayer(Layer* next_layer)
{
    ASSERT( next_layer, "NULL next layer" );

    nextLayer.emplace_back(next_layer);
}



/** ===============================================================================================
 * \name    Conv2D
 *
 * \brief   Construct a 2D convolution layer
 * 
 * \param   prev_layer          the pointer of previous layer
 * \param   input_size          [batch, channel, width, height]
 * \param   filter_size         [channel, width, height]
 * \param   activation_type     the activation type
 * \param   stride              [width, height]
 * \param   padding             [width, height]
 * 
 * \endcond
 * ================================================================================================
 */
Conv2D::Conv2D(Layer* prev_layer, int* input_size, int* filter_size, Activation_t activation_type, int* stride_value, int* padding_value)
        : Layer(prev_layer, Layer_t::Conv2D, input_size, filter_size, activation_type), stride(stride), padding(padding)
{
    init();
}


/** ===============================================================================================
 * \name    Conv2D
 *
 * \brief   Construct a 2D convolution as first layer
 * 
 * \param   input_size          [batch, channel, width, height]
 * \param   filter_size         [channel, width, height]
 * \param   activation_type     the activation type
 * \param   _stride             [width, height]
 * \param   _padding            [width, height]
 * 
 * \endcond
 * ================================================================================================
 */
Conv2D::Conv2D(int* input_size, int* filter_size, Activation_t activation_type, int* _stride, int* _padding)
        : Layer(nullptr, Layer_t::Conv2D, input_size, filter_size, activation_type), stride(_stride), padding(_padding)
{
    init();
}


/** ===============================================================================================
 * \name    Conv2D
 *
 * \brief   Construct a 2D convolution as first layer
 * 
 * \param   input_width         input width
 * \param   input_channel       input channel
 * \param   output_channel      output channel
 * \param   filter_width        the kernel width 
 * \param   activation_type     the activation type
 * \param   _stride             [width, height]
 * \param   _padding            [width, height]
 * 
 * \endcond
 * ================================================================================================
 */
Conv2D::Conv2D(int input_width, int input_channel, int output_channel, int filter_width, Activation_t activation_type, 
               int _stride, int _padding)
        : Layer(nullptr, Layer_t::Conv2D, input_width, input_channel, output_channel, filter_width, activation_type)
{
    stride  = new int[_stride, _stride];
    padding = new int[_padding, _padding];

    init();
}


/** ===============================================================================================
 * \name    Conv2D
 *
 * \brief   Construct a 2D convolution layer
 * 
 * \param   prev_layer          the pointer of previous layer
 * \param   filter_size         [channel, width, height]
 * \param   activation_type     the activation type
 * \param   _stride             [width, height]
 * \param   _padding            [width, height]
 * 
 * \endcond
 * ================================================================================================
 */
Conv2D::Conv2D(Layer* prev_layer, int* filter_size, Activation_t activation_type, int* _stride, int* _padding)
        : Layer(prev_layer, Layer_t::Conv2D, prev_layer->oFMapSize, filter_size, activation_type), stride(_stride), padding(_padding)
{
    init();
}


/** ===============================================================================================
 * \name    Conv2D
 *
 * \brief   Construct a 2D convolution layer
 * 
 * \param   prev_layer          the pointer of previous layer
 * \param   filter_size         [channel, width, height]
 * \param   activation_type     the activation type
 * \param   _stride             [width, height]
 * \param   _padding            [width, height]
 * 
 * \endcond
 * ================================================================================================
 */
Conv2D::Conv2D(Layer* prev_layer, int input_channel, int output_channel, int filter_width, Activation_t activation_type, int _stride, int _padding)
        : Layer(prev_layer, Layer_t::Conv2D, prev_layer->oFMapSize, input_channel, output_channel, filter_width, activation_type)
{
    stride  = new int[_stride, _stride];
    padding = new int[_padding, _padding];

    init();
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
    delete [] stride;
    delete [] padding;
}


/** ===============================================================================================
 * \name    init
 *
 * \brief   Initial the oFMapSize, defualt padding and stride is "1"
 * 
 * \endcond
 * 
 * The output feature map is calculated by the formula: floor{(iFMap - Kernel + 2 * padding) / stride} + 1
 * 
 * ================================================================================================
 */
void
Conv2D::init()
{
    oFMapSize = new int[4];
    oFMapSize[BATCH]    = iFMapSize[BATCH];
    oFMapSize[CHANNEL]  = filterSize[FILTER_CHANNEL_O];
    oFMapSize[HEIGHT]   = (double) (iFMapSize[HEIGHT] - filterSize[HEIGHT] + 2 * padding[HEIGHT]) / (double) (stride[HEIGHT]) + 1;
    oFMapSize[WIDTH]    = (double) (iFMapSize[WIDTH] - filterSize[WIDTH] + 2 * padding[WIDTH]) / (double) (stride[WIDTH]) + 1;
}



/** ===============================================================================================
 * \name    Pooling
 *
 * \brief   Construct a pooling layer
 * 
 * \param   prev_layer          the pointer of previous layer
 * \param   input_size          [batch, channel, width, height]
 * \param   filter_size         [channel, width, height]
 * \param   activation_type     the activation type
 * \param   _stride             [width, height]
 * \param   _padding            [width, height]
 * 
 * \endcond
 * ================================================================================================
 */
Pooling::Pooling(Layer* prev_layer, int* input_size, int* filter_size, Activation_t activation_type, int* _stride, int* _padding)
        : Conv2D(prev_layer, input_size, filter_size, activation_type, _stride, _padding)
{
    layerType = Layer_t::Pooling;
}


/** ===============================================================================================
 * \name    Pooling
 *
 * \brief   Construct a pooling layer
 * 
 * \param   prev_layer          the pointer of previous layer
 * \param   filter_size         [channel, width, height]
 * \param   _stride             [width, height]
 * \param   _padding            [width, height]
 * 
 * \endcond
 * ================================================================================================
 */
Pooling::Pooling(Layer* prev_layer, int* filter_size, int* _stride, int* _padding)
        : Conv2D(prev_layer, filter_size, Activation_t::None, _stride, _padding)
{
    layerType = Layer_t::Pooling;
}


/** ===============================================================================================
 * \name    Pooling
 *
 * \brief   Construct a pooling layer
 * 
 * \param   prev_layer          the pointer of previous layer
 * \param   filter_size         [channel, width, height]
 * \param   _stride             [width, height]
 * \param   _padding            [width, height]
 * 
 * \endcond
 * ================================================================================================
 */
Pooling::Pooling(Layer* prev_layer, int filter_width, int _stride, int _padding)
        : Conv2D(prev_layer, prev_layer->oFMapSize[CHANNEL], prev_layer->oFMapSize[CHANNEL], filter_width, Activation_t::None, _stride, _padding)
{
    layerType = Layer_t::Pooling;
}



/** ===============================================================================================
 * \name    Pooling
 *
 * \brief   Construct a pooling layer
 * 
 * \param   prev_layer          the pointer of previous layer
 * \param   input_size          [batch, channel, width, height]
 * \param   filter_size         [channel, width, height]
 * \param   activation_type     the activation type
 * \param   _stride             [width, height]
 * \param   _padding            [width, height]
 * 
 * \endcond
 * ================================================================================================
 */
Pooling::Pooling(Layer* prev_layer, int* input_size, int* filter_size, Activation_t activation_type, int* _stride, int* _padding)
        : Conv2D(prev_layer, input_size, filter_size, activation_type, _stride, _padding)
{
    layerType = Layer_t::Pooling;
}



/** ===============================================================================================
 * \name    Dense
 *
 * \brief   Construct a dense layer
 * 
 * \param   prev_layer          the pointer of previous layer
 * \param   input_size          [batch, channel, width, height]
 * \param   filter_size         [channel, width, height]
 * \param   activation_type     the activation type
 * 
 * \endcond
 * ================================================================================================
 */
Dense::Dense(Layer* prev_layer, int* input_size, int* filter_size, Activation_t activation_type)
        : Layer(prev_layer, Layer_t::Conv2D, input_size, filter_size, activation_type)
{
    init();
}


/** ===============================================================================================
 * \name    Dense
 *
 * \brief   Construct a dense layer
 * 
 * \param   prev_layer          the pointer of previous layer
 * \param   output_channel      output channel
 * \param   activation_type     the activation type
 * 
 * \endcond
 * ================================================================================================
 */
Dense::Dense(Layer* prev_layer, int output_channel, Activation_t activation_type)
        : Layer(prev_layer, Layer_t::Dense, 1
                , prev_layer->oFMapSize[WIDTH] * prev_layer->oFMapSize[HEIGHT] * prev_layer->oFMapSize[CHANNEL]
                , output_channel, 1, activation_type)
{
    init();
}


/** ===============================================================================================
 * \name    Dense
 *
 * \brief   Construct a dense layer
 * 
 * \param   prev_layer          the pointer of previous layer
 * \param   output_channel      output channel
 * \param   activation_type     the activation type
 * 
 * \endcond
 * ================================================================================================
 */
Dense::Dense(int input_channel, int output_channel, Activation_t activation_type)
        : Layer(NULL, Layer_t::Dense, 1, input_channel, output_channel, 1, activation_type)
{
    init();
}


/** ===============================================================================================
 * \name    init
 *
 * \brief   Initial the oFMapSize, defualt width and height is "1"
 * 
 * \endcond
 * 
 * ================================================================================================
 */
void
Conv2D::init()
{
    oFMapSize = new int[4];
    oFMapSize[BATCH]    = iFMapSize[BATCH];
    oFMapSize[CHANNEL]  = filterSize[FILTER_CHANNEL_O];
    oFMapSize[HEIGHT]   = 1;
    oFMapSize[WIDTH]    = 1;
}



