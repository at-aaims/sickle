#!/bin/bash
DIRNAME="${SLURM_JOBID}_Power"
mkdir -p $DIRNAME
if [ ! -d "$DIRNAME" ]; then
    echo "Directory $DIRNAME could not be created in $(pwd)"
    return
fi
if [ -z "${SLURM_NODEID}" ]; then
	FILEPREFIX=${HOSTNAME}
else
	FILEPREFIX=${SLURM_NODEID}
fi
{
echo ${HOSTNAME}
date
echo "-----"
rocm-smi -a --csv 2>&1
echo "-----"
echo "$HOSTNAME CPU Energy: $(cat /sys/cray/pm_counters/cpu_energy)" 2>&1
echo "$HOSTNAME Mem Energy: $(cat /sys/cray/pm_counters/memory_energy)"	 2>&1
echo "$HOSTNAME Accel0 Energy: $(cat /sys/cray/pm_counters/accel0_energy)" 2>&1
echo "$HOSTNAME Accel1 Energy: $(cat /sys/cray/pm_counters/accel1_energy)" 2>&1
echo "$HOSTNAME Energy: $(cat /sys/cray/pm_counters/accel2_energy)" 2>&1
echo "$HOSTNAME Accel3 Energy: $(cat /sys/cray/pm_counters/accel3_energy)" 2>&1
echo "$HOSTNAME Node Energy:   $(cat /sys/cray/pm_counters/energy)"
echo "-----"
} >> ${DIRNAME}/${FILEPREFIX}_energy 2>&1
