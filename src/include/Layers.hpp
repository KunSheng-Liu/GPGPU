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
#include "MMU.hpp"

/* ************************************************************************************************
 * Enumeration
 * ************************************************************************************************
 */

/* The index for "filterSize" */
#define FILTER_CHANNEL_O        0
#define FILTER_CHANNEL_I        1

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
    Tanh,
    Sigmoid,
    SoftMax,
    Max_Pool,
    Avg_Pool,
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

    Layer(char* = (char*)"None", vector<int> = {}, vector<int> = {}, char* = (char*)"None");

   ~Layer();

/* ************************************************************************************************
 * Type Define
 * ************************************************************************************************
 */
protected:
    /** ******************************************************************
     * \name    ThreadArg
     *
     * \brief   The pack of data pointer used in thread compile
     * 
     * \param   thread_id   for calculating the start and end position of loop
     * \param   num_thread  for calculating the start and end position of loop
     * \param   layer       the source layer pointer
     * \param   mmu         the memory management unit
     * \param   queue       the container to keep the compiled GPU requests
     * 
     * \endcond
     * ******************************************************************
     */
    struct ThreadArg
    {
        int threadID;
        int numThread;

        MMU* mmu;
        Layer* srcLayer;

        queue<Request*>* requestQueue;
        
        ThreadArg(int thread_id, int num_thread, Layer* layer, MMU* mmu, queue<Request*>* queue) 
            : threadID(thread_id), numThread(num_thread), srcLayer(layer), mmu(mmu), requestQueue(queue) {}
    };

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    
    /* virtual function */
    virtual void printInfo();
    virtual void changeBatch (int new_batch_size);
    virtual void memoryAllocate (MMU* mmu);
    PageRecord   memoryRelease  (MMU* mmu);

    /* Make the kernel dependency by layer graph */
    virtual vector<Kernel*> compileToKernel (int app_id, int model_id, vector<Kernel>& container, vector<Kernel*> dependency);

    /* Compile the current layer graph into GPU command */
    void Compile (MMU* mmu, Kernel* targetKernel);

private:
    /* A thread wrapper */
    static void* threadCompile (void* arg);

    /* pure virtual function */
    virtual void issueLayer(ThreadArg* threadArg) = 0;
    virtual void calculateOFMapSize() = 0;


/* ************************************************************************************************
 * Basic parameter I/O
 * ************************************************************************************************
 */
public:
    /* Layer data I/O */
    virtual void setIFMap  (pair<int, vector<unsigned char>*> data);
    virtual void setFilter (pair<int, vector<unsigned char>*> data);

    int getMemoryUsage();
    vector<int> getIFMapSize  (void) const {return iFMapSize;}
    vector<int> getOFMapSize  (void) const {return oFMapSize;}
    vector<int> getFilterSize (void) const {return filterSize;}
    pair<int, vector<unsigned char>*> getOFMap  (void) {return oFMap;}
    pair<int, vector<unsigned char>*> getIFMap  (void) {return iFMap;}
    pair<int, vector<unsigned char>*> getFilter (void) {return filter;}


/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
public:
    /* The index of layer. Each layer have a unique index */
    const int layerID;

    /* Type of layer */
    const char* layerType;
    
    /* The activation type */
    const char* activationType;

protected:

    /* Number of layer be created */
    static int layerCount;

    static int vaCount;

    /* The dimensions of feature map and filter */
    vector<int> iFMapSize;     // In order "batch", "channel", "height", and "width"
    vector<int> oFMapSize;     // In order "batch", "channel", "height", and "width"
    vector<int> filterSize;    // In order "FILTER_CHANNEL_I", "FILTER_CHANNEL_O", "height", and "width"

    /* The array of feature map and filter in byte format */
    pair<int, vector<unsigned char>*> iFMap;       // Reference to input data
    pair<int, vector<unsigned char>*> oFMap;       // Output data, create by instanced layer
    pair<int, vector<unsigned char>*> filter;      // Reference to filter data
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

    Conv2D(char*, vector<int> = {}, vector<int> = {}, char* = (char*)"None", vector<int> = {}, vector<int> = {});

    Conv2D(vector<int> = {}, vector<int> = {}, char* = (char*)"None", vector<int> = {}, vector<int> = {});

    Conv2D(vector<int> = {}, vector<int> = {}, char* = (char*)"None", int = 1, int = 0);

    ~Conv2D();

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    // void memoryAllocate(MMU* mmu) override;
    void printInfo() override;
    void issueLayer(ThreadArg* threadArg) override;
    
private:
    void calculateOFMapSize() override;

/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
protected:
    vector<int> stride;
    vector<int> padding;
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

    Pooling(vector<int> = {}, vector<int> = {}, char* = (char*)"None", vector<int> = {}, vector<int> = {});

    Pooling(vector<int> = {}, vector<int> = {}, char* = (char*)"None", int = 0, int = 0);

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    void issueLayer(ThreadArg* threadArg) override;

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

    Flatten(vector<int> = {});

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    void printInfo() override;
    void issueLayer(ThreadArg* threadArg) override;
    
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

    ByPass(vector<int>);
    ByPass(vector<int>, vector<int>);

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    void printInfo() override;
    void issueLayer(ThreadArg* threadArg) override;
    
private:
    void calculateOFMapSize() override {};

};



/** ===============================================================================================
 * \name    Dense
 * 
 * \brief   A Dense layer inherits the \b "Layer" class. 
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

    Dense(vector<int> = {}, vector<int> = {}, char* = (char*)"None");

    Dense(vector<int>, int);

    // Dense(int, int, char* = (char*)"None");


/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    void printInfo() override;
    void issueLayer(ThreadArg* threadArg) override;
    
private:
    void calculateOFMapSize() override;

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */

};


#endif