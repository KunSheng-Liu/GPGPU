/**
 * \name    GPGPU.hpp
 * 
 * \brief   Implement the GPGPU
 * 
 * \date    APR 18, 2023
 */
#include "include/GPGPU.hpp"

/* ************************************************************************************************
 * Global Variable
 * ************************************************************************************************
 */
/* Statistic */
unsigned long long total_gpu_cycle = 0;

/* Thread Protect */
pthread_mutex_t* ioMutex = new pthread_mutex_t;

/** ===============================================================================================
 * \name    GPGPU
 * 
 * \brief   Instance the used modules
 * 
 * \endcond
 * ================================================================================================
 */
GPGPU::GPGPU() : mMC(MemoryController(system_resource.DRAM_SPACE + system_resource.VRAM_SPACE, PAGE_SIZE)), mGPU(GPU(&mMC)), mCPU(CPU(&mMC, &mGPU))
{
    mGMMU = mGPU.getGMMU();

	std::cout << program_name << std::endl;

	ofstream file(LOG_OUT_PATH + program_name + ".txt", std::ofstream::out | std::ofstream::trunc);
	file.close();
}


/** ===============================================================================================
 * \name    GPGPU
 * 
 * \brief   Destruct GPGPU
 * 
 * \endcond
 * ================================================================================================
 */
GPGPU::~GPGPU()
{

}


/** ===============================================================================================
 * \name    run
 * 
 * \brief   Start to simulating
 * 
 * \endcond
 * ================================================================================================
 */
void 
GPGPU::run ()
{
	timeval start, end;
    bool Finish = false;
	while (!Finish)
    {       
        int clock_mask = next_clock_domain();

		if (clock_mask & MC_MASK) {
#if (PRINT_TIME_STEP)
			gettimeofday(&start, NULL);
#endif
			mMC.cycle();
#if (PRINT_TIME_STEP)
			gettimeofday(&end, NULL);
			std::cout << "MC cycle spend time: " << to_string((1000000 * (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)) * 0.001) << " ms" << std::endl;
#endif

		}
		if (clock_mask & GMMU_MASK) {
#if (PRINT_TIME_STEP)
			gettimeofday(&start, NULL);
#endif
			mGMMU->cycle();
#if (PRINT_TIME_STEP)
			gettimeofday(&end, NULL);
			std::cout << "GMMU cycle spend time: " << to_string((1000000 * (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)) * 0.001) << " ms" << std::endl;
#endif
		}
		if (clock_mask & GPU_MASK) {
#if (PRINT_TIME_STEP)
			gettimeofday(&start, NULL);
#endif
			mGPU.cycle();
#if (PRINT_TIME_STEP)
			gettimeofday(&end, NULL);
			std::cout << "GPU cycle spend time: " << to_string((1000000 * (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)) * 0.001) << " ms" << std::endl;
#endif
			if (++total_gpu_cycle % 10000 == 0) std::cout << total_gpu_cycle << std::endl;
		}
		if (clock_mask & CPU_MASK) {
#if (PRINT_TIME_STEP)
			gettimeofday(&start, NULL);
#endif
			mCPU.cycle();
#if (PRINT_TIME_STEP)
			gettimeofday(&end, NULL);
			std::cout << "CPU cycle spend time: " << to_string((1000000 * (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)) * 0.001) << " ms" << std::endl;
#endif
		}
        Finish = mCPU.Check_All_Applications_Finish();
    }
}


/** ===============================================================================================
 * \name    next_clock_domain
 * 
 * \brief   Determine wheather the module needs to be executed
 * 
 * \endcond
 * ================================================================================================
 */
int 
GPGPU::next_clock_domain() 
{  
	long double smallest = min4(gpu_time, cpu_time, mc_time, gmmu_time);

	int mask = 0x00;
	if (gpu_time <= smallest) {
		mask |= GPU_MASK;
		gpu_time += gpu_period;
	}
	if (cpu_time <= smallest) {
		mask |= CPU_MASK;
		cpu_time += cpu_period;
	}
	if (gmmu_time <= smallest) {
		mask |= GMMU_MASK;
		gmmu_time += gmmu_period;
	}
	if (mc_time <= smallest) {
		mask |= MC_MASK;
		mc_time += mc_period;
	}

	return mask;
}