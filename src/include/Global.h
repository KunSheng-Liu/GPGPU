/** 
 * \name    Global.h
 * 
 * \brief   Declare all configure
 * 
 * \date    Mar 31, 2023
 */

#ifndef _GLOBAL_H_
#define _GLOBAL_H_

/* ************************************************************************************************
 * Type Define
 * ************************************************************************************************
 */
struct RuntimeInfo
{
    // Size of GPU batch processing
	float batch_size = 0;
	unsigned PF_times = 0;
	unsigned PF_pages = 0;
	unsigned n_previous_access_page = 0;

    std::list<int> SM_List;
};

/* ************************************************************************************************
 * Enumeration
 * ************************************************************************************************
 */
typedef enum {
	Baseline,
	Balance,
	SMD,
	R_SMD,
}SM_Dispatch;

typedef enum {
    None,
	MEMA,
	R_MEMA,
}MEM_Allocate;


/* ************************************************************************************************
 * Global variable
 * ************************************************************************************************
 */
extern SM_Dispatch  SM_MODE;
extern MEM_Allocate MEM_MODE;
extern unsigned long long total_gpu_cycle;


#endif