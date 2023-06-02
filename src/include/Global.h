
#ifndef _GLOBAL_H_
#define _GLOBAL_H_

#include <fstream>
#include <list>

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
	Greedy, Baseline,
	BARM, LazyB,
	My
}SCHEDULER;
typedef enum {
    SEQUENTIAL, PARALLEL,
}INFERENCE_TYPE;

typedef enum {
    DISABLE, MAX,
}BATCH_METHOD;

typedef enum {
    None, Average,
	MEMA, R_MEMA,
}MEM_ALLOCATION;

typedef enum {
    LeNet, CaffeNet, ResNet18, GoogleNet, VGG16,
    ALL,
    LIGHT, HEAVY, MIX,
    TEST1, TEST2,
}APPLICATION;

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

	RuntimeRecord& operator+= (const RuntimeRecord& other) {
		PF_times += other.PF_times;
		PF_pages += other.PF_pages;
		// batch_process_size = PF_pages / PF_times;
		return *this;
	}
};

struct Command {
    SCHEDULER    	SCHEDULER_MODE;
    INFERENCE_TYPE 	INFERENCE_MODE;
    BATCH_METHOD   	BATCH_MODE;
    MEM_ALLOCATION 	MEM_MODE;
    std::list<std::pair<APPLICATION, int>> TASK_LIST;

    Command() : SCHEDULER_MODE(Greedy), INFERENCE_MODE(SEQUENTIAL), BATCH_MODE(DISABLE), MEM_MODE(None) {}
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