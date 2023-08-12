
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
	Baseline,
	Average,
	BARM,
	SALBI
}SCHEDULER;

typedef enum {
    DISABLE, MAX,
}BATCH_METHOD;

typedef enum {
    LeNet, CaffeNet, ResNet18, GoogleNet, VGG16, SqueezeNet,
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
    BATCH_METHOD   	BATCH_MODE;
    std::list<std::pair<APPLICATION, std::tuple<int /*batch_size*/, float /*arrival_time*/, float /*period*/, float /*deadline*/>>> TASK_LIST;

    Command() : SCHEDULER_MODE(Baseline), BATCH_MODE(MAX) {}
};

struct Resource {
    unsigned long long	SM_NUM;
    unsigned long long 	DRAM_SPACE;
    unsigned long long 	VRAM_SPACE;
	unsigned long long 	DEADLINE_CYCLE;

    Resource() : SM_NUM(8), DRAM_SPACE((unsigned long long) 32 * 1024 * 1024 * 1024), VRAM_SPACE(-1), DEADLINE_CYCLE(-1) {}
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

extern Resource	system_resource;

extern std::string program_name;

#endif