/**
 * \name    Layer.h
 * 
 * \brief   Declare the layer API 
 *          
 * \note    Available model type:
 *          - \b Conv2D
 *          - \b Dense 
 *          - \b Flatten 
 *          - \b Pooling 
 *          - \b Inception 
 * 
 * \date    Mar 31, 2023
 */

#ifndef _LAYER_H_
#define _LAYER_H_

/* ************************************************************************************************
 * Include Library
 * ************************************************************************************************
 */
#include "App_config.h"
#include "Log.h"


/* ************************************************************************************************
 * Enumeration
 * ************************************************************************************************
 */

/* The index for "iFMapSize" and "oFMapSize" */
#define BATCH               0
#define CHANNEL             1
#define HEIGHT              2
#define WIDTH               3

/* The index for "filterSize" */
#define FILTER_CHANNEL_I    0
#define FILTER_CHANNEL_O    1

/* All avaliable layer type*/
enum class Layer_t{
    None        = -1,
    Conv2D      =  0,
    Dense       =  1,
    Flatten     =  2,
    Pooling     =  3,
    Inception   =  4,
};

/* All avaliable activation type*/
enum class Activation_t{
    None        = -1,
    ReLU        =  0,
    Sigmoid     =  1,
};


/** ===============================================================================================
 * \name    Layer
 * 
 * \brief   The base class of NN layer. You can add new layer type by inheritance this class and
 *          override the virtual function to fit the desire.
 * \endcond
 * ================================================================================================
 */
class Layer
{
/* ************************************************************************************************
 * Class Constructor
 * ************************************************************************************************
 */ 
public:

    Layer(Layer*, Layer_t, int*, int*, Activation_t = Activation_t::None);

    Layer(Layer*, Layer_t, int*, int, int, int, Activation_t = Activation_t::None);
    
    Layer(Layer*, Layer_t, int, int, int, int, Activation_t = Activation_t::None);

    // Layer(Layer_t, int, int); 
    
    ~Layer();

/* ************************************************************************************************
 * Type Define
 * ************************************************************************************************
 */


/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    /* basic parameter functions */
    void setLayerIndex (int index) {layerIndex = index;}

    int  getLayerIndex (void) {return layerIndex;}

    /* setup layer dependency */
    void addPrevLayer (Layer*);
    void addNextLayer (Layer*);
    void setUpConnection (Layer*);

    /* layer stats */
    bool isExecuting   (void) {return flagExecuting;}
    bool isFinish      (void) {return flagFinish;}

    /*  */
    virtual void PrintInfo();

private:
    /* Initial layer */
    void init ();


/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
public:
    /* The index of layer. Each layer have a unique index */
    int layerIndex;

    /* Type of layer */
    Layer_t layerType;
    
    /* The activation type */
    Activation_t activationType;

    /* Layer stats flags */
    bool flagExecuting;
    bool flagFinish;

    /* The dimensions of feature map and filter */
    int* iFMapSize;     // In order "batch", "channel", "width", and "height"
    int* oFMapSize;     // In order "width", "height", "channel", and "batch"
    int* filterSize;    // In order "width", "height", "channel depth", and "number of channel"

    /* The feature map and filter */
    unsigned char* iFMap;
    unsigned char* oFMap;
    unsigned char* filter;

    /* Record the merge in and branch out layers */
    std::vector <Layer*> prevLayer;
    std::vector <Layer*> nextLayer;
};



/** ===============================================================================================
 * \name    Conv2D
 * 
 * \brief   A 2D convolution layer inherits the \b "Layer" class. 
 * 
 * \endcond
 * ================================================================================================
 */
class Conv2D: public Layer
{
/* ************************************************************************************************
 * Class Constructor
 * ************************************************************************************************
 */ 
public:

    Conv2D(Layer*, int*, int*, Activation_t = Activation_t::None, int* = NULL, int* = NULL);

    Conv2D(int*, int*, Activation_t = Activation_t::None, int* = NULL, int* = NULL);

    Conv2D(int, int, int ,int , Activation_t = Activation_t::None, int = 1, int = 0);

    Conv2D(Layer*, int*, Activation_t = Activation_t::None, int* = NULL, int* = NULL);

    Conv2D(Layer*, int, int ,int , Activation_t = Activation_t::None, int = 1, int = 0);

    ~Conv2D();

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
private:
    void init ();

/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
private:
    int* stride;
    int* padding;
};



/** ===============================================================================================
 * \name    Pooling
 * 
 * \brief   A pooling layer inherits the \b "Conv2D" class. 
 * 
 * \endcond
 * ================================================================================================
 */
class Pooling: public Conv2D
{
/* ************************************************************************************************
 * Class Constructor
 * ************************************************************************************************
 */ 
public:

    Pooling(Layer*, int*, int*, Activation_t = Activation_t::None, int* = NULL, int* = NULL);

    Pooling(Layer*, int*, int* = NULL, int* = NULL);

    Pooling(Layer*, int, int = 0, int = 0);
};



/** ===============================================================================================
 * \name    Dense
 * 
 * \brief   A pooling layer inherits the \b "Layer" class. 
 * 
 * \endcond
 * ================================================================================================
 */
class Dense: public Layer
{
/* ************************************************************************************************
 * Class Constructor
 * ************************************************************************************************
 */ 
public:

    Dense(Layer*, int*, int*, Activation_t = Activation_t::None);

    Dense(Layer*, int, Activation_t = Activation_t::None);

    Dense(int, int, Activation_t = Activation_t::None);

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
private:
    void init ();
};



#endif