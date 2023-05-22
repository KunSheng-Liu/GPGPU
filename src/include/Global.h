
#ifndef _GLOBAL_H_
#define _GLOBAL_H_

#include "fstream"

/* ************************************************************************************************
 * Enumeration
 * ************************************************************************************************
 */
typedef enum {
    SEQUENTIAL,
	PARALLEL,

	INFERENCE_TYPE_END
}INFERENCE_TYPE;

typedef enum {
    DISABLE,
	MAX,

	BATCH_METHOD_END
}BATCH_METHOD;

typedef enum {
	Greedy,
	Baseline,
	Equal,
	SMD,
	R_SMD,

	SM_DISPATCH_END
}SM_DISPATCH;

typedef enum {
    None,
	Average,
	MEMA,
	R_MEMA,

	MEM_ALLOCATION_END
}MEM_ALLOCATION;

typedef enum {
    LIGHT,
    HEAVY,
    MIX,
    LeNet,
	CaffeNet,
    ResNet18,
    VGG16,
    GoogleNet,
    ALL,
    TEST1,
    TEST2,

	TASK_SET_END
}TASK_SET;

/* ************************************************************************************************
 * Type Define
 * ************************************************************************************************
 */
struct RuntimeRecord
{
    /* Size of GPU batch processing */
	float batch_process_size = 0;
	unsigned PF_times = 0;
	unsigned PF_pages = 0;

	// /* Operators */
	// RuntimeRecord operator+ (const RuntimeRecord& other) const {
	// 	return {(PF_pages + other.PF_pages) / (PF_times + other.PF_pages)
	// 	, PF_times + other.PF_times, PF_pages + other.PF_pages};}

	// RuntimeRecord& operator+= (const RuntimeRecord& other) {
	// 	PF_times += other.PF_times;
	// 	PF_pages += other.PF_pages;
	// 	batch_process_size = PF_pages / PF_times;
	// 	return *this;
	// }
};

struct Command {
    INFERENCE_TYPE INFERENCE_MODE;
    BATCH_METHOD   BATCH_MODE;
    SM_DISPATCH    SM_MODE;
    MEM_ALLOCATION MEM_MODE;
    TASK_SET 	   TASK_MODE;

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