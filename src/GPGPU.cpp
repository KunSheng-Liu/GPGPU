/**
 * \name    GPGPU.hpp
 * 
 * \brief   Implement the GPGPU
 * 
 * \date    APR 18, 2023
 */
#include "include/GPGPU.hpp"


/** ===============================================================================================
 * \name    GPGPU
 * 
 * \brief   Instance the used modules
 * 
 * \endcond
 * ================================================================================================
 */
GPGPU::GPGPU()
{
    mMC  = new MemoryControl(DISK_SPACE, PAGE_SIZE);
    mCPU = new CPU(mMC);
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
            cout << "Memory Controller" << endl;
			// m_mc.cycle();
		}
		if (clock_mask & GMMU_MASK) {
            cout << "GMMU" << endl;
			// m_gmmu.cycle();
		}
		if (clock_mask & GPU_MASK) {
            cout << "GPU" << endl;
			// m_gpu.cycle();
			
		}
		if (clock_mask & CPU_MASK) {
            cout << "CPU" << endl;
			mCPU->cycle();
		}

        // Finish = m_cpu.check_all_application_finished() & m_mc.check_finished();
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