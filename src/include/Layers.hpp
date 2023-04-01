/**
 * \name    Layer.hpp
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

#ifndef _LAYER_HPP_
#define _LAYER_HPP_

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

/* The index for "filterSize" */
#define FILTER_CHANNEL_I    0
#define FILTER_CHANNEL_O    1

/* The index for "iFMapSize" and "oFMapSize" */
#define BATCH               0
#define CHANNEL             1
#define HEIGHT              2
#define WIDTH               3

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

    Layer(Layer_t = Layer_t::None, int* = NULL, int* = NULL, int* = NULL, Activation_t = Activation_t::None);

    // Layer(Layer_t, int*, int, int, int, Activation_t = Activation_t::None);
    
    // Layer(Layer_t, int, int, int, int, Activation_t = Activation_t::None);

   ~Layer();

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    /* pure virtual function */
    virtual void issueLayer() = 0;
    virtual void printInfo() = 0;

private:
    /* pure virtual function */
    virtual int* calculateOFMapSize() = 0;

/* ************************************************************************************************
 * Basic parameter I/O
 * ************************************************************************************************
 */
public:
    /* basic parameter I/O */
    void setLayerIndex (int index) {layerIndex = index;}
    int  getLayerIndex (void) {return layerIndex;}

    /* Layer data I/O */
    void setiFMap  (unsigned char* data) {iFMap = data;}
    void setoFMap  (unsigned char* data) {oFMap = data;}
    void setFilter (unsigned char* data) {filter = data;}

    unsigned char* getiFMap (void) {return iFMap;}
    unsigned char* getoFMap (void) {return oFMap;}
    unsigned char* getFilter (void) {return filter;}

    /* layer stats */
    bool isExecuting   (void) {return flagExecuting;}
    bool isFinish      (void) {return flagFinish;}


/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
public:
    /* Type of layer */
    const Layer_t layerType;
    
    /* The activation type */
    const Activation_t activationType;

    /* The dimensions of feature map and filter */
    int* oFMapSize;           // In order "batch", "channel", "height", and "width"
    const int* iFMapSize;     // In order "batch", "channel", "height", and "width"
    const int* filterSize;    // In order "FILTER_CHANNEL_I", "FILTER_CHANNEL_O", "height", and "width"

protected:
    /* The index of layer. Each layer have a unique index */
    int layerIndex;

    /* Layer stats flags */
    bool flagExecuting;
    bool flagFinish;

    /* The array of feature map and filter in byte format */
    unsigned char* iFMap;
    unsigned char* oFMap;
    unsigned char* filter;
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

    Conv2D(int*, int*, Activation_t = Activation_t::None, int* = NULL, int* = NULL);

    Conv2D(Layer_t, int*, int*, Activation_t = Activation_t::None, int* = NULL, int* = NULL);

    // Conv2D(int*, int*, Activation_t = Activation_t::None, int* = NULL, int* = NULL);

    // Conv2D(int, int, int ,int , Activation_t = Activation_t::None, int = 1, int = 0);

    // Conv2D(Layer*, int*, Activation_t = Activation_t::None, int* = NULL, int* = NULL);

    // Conv2D(Layer*, int, int ,int , Activation_t = Activation_t::None, int = 1, int = 0);

    ~Conv2D();

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    void issueLayer() override;
    void printInfo() override;
    
private:
    int* calculateOFMapSize() override;

/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
private:
    const int* stride;
    const int* padding;
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

    Pooling(int*, int*, Activation_t = Activation_t::None, int* = NULL, int* = NULL);

    // Pooling(Layer*, int*, int* = NULL, int* = NULL);

    // Pooling(Layer*, int, int = 0, int = 0);
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

    Dense(int*, int*, Activation_t = Activation_t::None);

    // Dense(Layer*, int, Activation_t = Activation_t::None);

    // Dense(int, int, Activation_t = Activation_t::None);


/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    void issueLayer() override;
    void printInfo() override;
    
private:
    int* calculateOFMapSize() override;

};



#endif