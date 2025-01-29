#!/bin/bash
DIRNAME="${SLURM_JOBID}_Power"
mkdir -p $DIRNAME
if [ ! -d "$DIRNAME" ]; then
	echo "Directory $DIRNAME could not be created in $(pwd)"
	return	
fi
while true
do
#    rocm-smi -a --csv 2>&1 >> $DIRNAME/rocm-smi_${SLURM_NODEID}.txt &
    cat /sys/cray/pm_counters/cpu_power >> $DIRNAME/CRAYPM_cpu_${SLURM_NODEID}.txt &
    cat /sys/cray/pm_counters/memory_power >> $DIRNAME/CRAYPM_mem_${SLURM_NODEID}.txt &
    cat /sys/cray/pm_counters/accel0_power >> $DIRNAME/CRAYPM_accl0_${SLURM_NODEID}.txt &
    cat /sys/cray/pm_counters/accel1_power >> $DIRNAME/CRAYPM_accl1_${SLURM_NODEID}.txt &
    cat /sys/cray/pm_counters/accel2_power >> $DIRNAME/CRAYPM_accl2_${SLURM_NODEID}.txt &
    cat /sys/cray/pm_counters/accel3_power >> $DIRNAME/CRAYPM_accl3_${SLURM_NODEID}.txt &
    cat /sys/cray/pm_counters/power        >> $DIRNAME/CRAYPM_pwr_${SLURM_NODEID}.txt &
    sleep 1
done
