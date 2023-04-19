/**
 * \name    Layers.hpp
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

#ifndef _LAYERS_HPP_
#define _LAYERS_HPP_

/* ************************************************************************************************
 * Include Library
 * ************************************************************************************************
 */
#include "App_config.h"
#include "Log.h"

#include "Kernel.hpp"
#include "MemoryControl.hpp"
#include "MMU.hpp"

/* ************************************************************************************************
 * Enumeration
 * ************************************************************************************************
 */

/* The index for "filterSize" */
#define FILTER_CHANNEL_I        0
#define FILTER_CHANNEL_O        1

/* The index for "stride" and "padding" */
#define STRIDE_PADDING_HEIGHT   0
#define STRIDE_PADDING_WIDTH    1

/* The index for "iFMapSize" and "oFMapSize" */
#define BATCH                   0
#define CHANNEL                 1
#define HEIGHT                  2
#define WIDTH                   3

/* All avaliable layer type*/
enum class Layer_t{
    None,
    ByPass,
    Conv2D,
    Dense,
    Flatten,
    Inception,
    Pooling,
};

/* All avaliable activation type*/
enum class Activation_t{
    None,
    ReLU,
    Sigmoid1,
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

    Layer(char* = (char*)"None", vector<int>* = nullptr, vector<int>* = nullptr, char* = (char*)"None");

   ~Layer();

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    /* virtual function */
    virtual void printInfo();
    virtual void memoryAllocate (MMU* mmu);
    virtual void changeBatch (int new_batch_size);
    /* pure virtual function */
    virtual vector<Kernel*> issueLayer(vector<Kernel>& container, vector<Kernel*> dependency) = 0;

private:
    /* pure virtual function */
    virtual void calculateOFMapSize() = 0;

/* ************************************************************************************************
 * Basic parameter I/O
 * ************************************************************************************************
 */
public:
    /* basic parameter I/O */
    int  getLayerIndex (void) {return layerIndex;}

    /* Layer data I/O */
    virtual void setIFMap  (vector<unsigned char>* data);
    virtual void setFilter (vector<unsigned char>* data);

    vector<int>* getIFMapSize  (void) {return iFMapSize;}
    vector<int>* getOFMapSize  (void) {return oFMapSize;}
    vector<int>* getFilterSize (void) {return filterSize;}
    vector<unsigned char>* getOFMap  (void) {return oFMap;}
    vector<unsigned char>* getIFMap  (void) {return iFMap;}
    vector<unsigned char>* getFilter (void) {return filter;}

    /* layer stats */
    bool isExecuting   (void) {return flagExecuting;}
    bool isFinish      (void) {return flagFinish;}


/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
public:
    /* The index of layer. Each layer have a unique index */
    const int layerIndex;

    /* Type of layer */
    char* layerType;
    
    /* The activation type */
    char* activationType;

protected:

    /* Number of layer be created */
    static int layerCount;

    /* Layer stats flags */
    bool flagExecuting;
    bool flagFinish;

    /* The dimensions of feature map and filter */
    vector<int>* iFMapSize;     // In order "batch", "channel", "height", and "width"
    vector<int>* oFMapSize;     // In order "batch", "channel", "height", and "width"
    vector<int>* filterSize;    // In order "FILTER_CHANNEL_I", "FILTER_CHANNEL_O", "height", and "width"

    /* The array of feature map and filter in byte format */
    vector<unsigned char>* iFMap;       // Reference to input data
    vector<unsigned char>* oFMap;       // Output data, create by instanced layer
    vector<unsigned char>* filter;      // Reference to filter data
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

    Conv2D(char*, vector<int>* = nullptr, vector<int>* = nullptr, char* = (char*)"None", vector<int>* = {}, vector<int>* = {});

    Conv2D(vector<int>* = nullptr, vector<int>* = nullptr, char* = (char*)"None", vector<int>* = {}, vector<int>* = {});

    Conv2D(vector<int>* = nullptr, vector<int>* = nullptr, char* = (char*)"None", int = 1, int = 0);

    ~Conv2D();

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    // void memoryAllocate(MMU* mmu) override;
    void printInfo() override;
    vector<Kernel*> issueLayer(vector<Kernel>& container, vector<Kernel*> dependency) override;
    
private:
    void calculateOFMapSize() override;

/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
protected:
    vector<int>* stride;
    vector<int>* padding;
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

    Pooling(vector<int>* = nullptr, vector<int>* = nullptr, char* = (char*)"None", vector<int>* = {}, vector<int>* = {});

    Pooling(vector<int>* = nullptr, vector<int>* = nullptr, char* = (char*)"None", int = 0, int = 0);

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    vector<Kernel*> issueLayer(vector<Kernel>& container, vector<Kernel*> dependency) override;

};



/** ===============================================================================================
 * \name    Flatten
 * 
 * \brief   A flatten layer inherits the \b "Layer" class. 
 * 
 * \endcond
 * ================================================================================================
 */
class Flatten: public Layer
{
/* ************************************************************************************************
 * Class Constructor
 * ************************************************************************************************
 */ 
public:

    Flatten(vector<int>*);

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    void printInfo() override;
    vector<Kernel*> issueLayer(vector<Kernel>& container, vector<Kernel*> dependency) override;
    
private:
    void calculateOFMapSize() override;

};



/** ===============================================================================================
 * \name    ByPass
 * 
 * \brief   A flatten layer inherits the \b "Layer" class. 
 * 
 * \endcond
 * ================================================================================================
 */
class ByPass: public Layer
{
/* ************************************************************************************************
 * Class Constructor
 * ************************************************************************************************
 */ 
public:

    ByPass(vector<int>*);

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    void printInfo() override;
    vector<Kernel*> issueLayer(vector<Kernel>& container, vector<Kernel*> dependency) override;
    
private:
    void calculateOFMapSize() override;

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

    Dense(vector<int>*, vector<int>*, char* = (char*)"None");

    Dense(vector<int>*, int);

    // Dense(int, int, char* = (char*)"None");


/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    void printInfo() override;
    vector<Kernel*> issueLayer(vector<Kernel>& container, vector<Kernel*> dependency) override;
    
private:
    void calculateOFMapSize() override;

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
private:
    int outWidth;

};


#endif