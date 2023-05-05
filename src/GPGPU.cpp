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
unsigned long long total_gpu_cycle = 0;
SM_Dispatch  SM_MODE  = SM_Dispatch::Baseline;
MEM_Allocate MEM_MODE = MEM_Allocate::None;

/** ===============================================================================================
 * \name    GPGPU
 * 
 * \brief   Instance the used modules
 * 
 * \endcond
 * ================================================================================================
 */
GPGPU::GPGPU() : mMC(MemoryController(DISK_SPACE, PAGE_SIZE)), mGPU(GPU(&mMC)), mCPU(CPU(&mMC, &mGPU))
{
    mGMMU = mGPU.getGMMU();
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
    bool Finish = false;
	while (!Finish)
    {       
        int clock_mask = next_clock_domain();

		if (clock_mask & MC_MASK) {
			mMC.cycle();
		}
		if (clock_mask & GMMU_MASK) {
			mGMMU->cycle();
		}
		if (clock_mask & GPU_MASK) {
			mGPU.cycle();
			total_gpu_cycle++;
			
		}
		if (clock_mask & CPU_MASK) {
			mCPU.cycle();
		}

    	// ASSERT(total_gpu_cycle != 150);
        Finish = mCPU.Check_All_Applications_Finish();
    }
	while(1){};
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
GPGPU::next_clock_domain() {
    
	double smallest = min4(gpu_time, cpu_time, mc_time, gmmu_time);

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