
#ifndef _GLOBAL_H_
#define _GLOBAL_H_

#include "fstream"

/* ************************************************************************************************
 * Enumeration
 * ************************************************************************************************
 */
/* Logger level */
#define LOG_OFF                             0
#define ERROR                               1
#define WARNNING                            2
#define INFO                                3
#define DEBUG                               4
#define TRACE                               5
#define VERBOSE                             6
#define LOG_ALL                             7

/* Policy*/
typedef enum {
	Greedy,
	Baseline,
	BARM,
	LazyB,
}SCHEDULER;
typedef enum {
    SEQUENTIAL,
	PARALLEL,
}INFERENCE_TYPE;

typedef enum {
    DISABLE,
	MAX,
}BATCH_METHOD;

typedef enum {
    None,
	Average,
	MEMA,
	R_MEMA,
}MEM_ALLOCATION;

typedef enum {
    LeNet, CaffeNet, ResNet18, GoogleNet, VGG16,
    ALL,
    LIGHT, HEAVY, MIX,
    TEST1, TEST2,
}TASK_SET;

/* ************************************************************************************************
 * Type Define
 * ************************************************************************************************
 */
struct RuntimeRecord
{
	unsigned long long start_time, end_time;

    /* Size of GPU batch processing */
	float batch_process_size = 0;
	unsigned PF_times = 0;
	unsigned PF_pages = 0;

	// /* Operators */
	// RuntimeRecord operator+ (const RuntimeRecord& other) const {
	// 	return {(PF_pages + other.PF_pages) / (PF_times + other.PF_pages)
	// 	, PF_times + other.PF_times, PF_pages + other.PF_pages};}

	RuntimeRecord& operator+= (const RuntimeRecord& other) {
		PF_times += other.PF_times;
		PF_pages += other.PF_pages;
		// batch_process_size = PF_pages / PF_times;
		return *this;
	}
};

struct Command {
    INFERENCE_TYPE 	INFERENCE_MODE;
    BATCH_METHOD   	BATCH_MODE;
    SCHEDULER    	SM_MODE;
    MEM_ALLOCATION 	MEM_MODE;
    TASK_SET 	   	TASK_MODE;

    Command() : INFERENCE_MODE(SEQUENTIAL), BATCH_MODE(DISABLE), SM_MODE(Greedy), MEM_MODE(None), TASK_MODE(LIGHT) {}
};

/* ************************************************************************************************
 * Global variable
 * ************************************************************************************************
 */
/* Statistic */
extern unsigned long long total_gpu_cycle;

/* Thread Protect */
extern pthread_mutex_t* ioMutex;

/* Command */
extern Command command;

extern std::string program_name;

#endif