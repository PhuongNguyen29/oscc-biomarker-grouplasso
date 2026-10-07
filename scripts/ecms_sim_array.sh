#!/bin/bash
# ==============================================================================
# ECMS simulation: SGE array job for the HPC (format follows run_step2_ig_job5.sh).
# One array task = one modulePreservation call. Tasks whose output already exists
# exit at once, so resubmitting the same command is safe.
#
# Submit FROM THE PROJECT ROOT (the folder that contains scripts/ and results/):
#   mkdir -p results/ecms_sim/sge_logs
#   qsub -t 1-100   scripts/ecms_sim_array.sh B null
#   qsub -t 1-1000  scripts/ecms_sim_array.sh B data
#   qsub -t 1-150   scripts/ecms_sim_array.sh B ownnull     # 5 replicates x 30
#
# NO e-mail flags (-m / -M) on purpose: an array of 1,000 tasks would send up to
# 3,000 e-mails. Check progress with qstat or the log folder instead.
# ==============================================================================
#$ -N ecms_sim
#$ -cwd
#$ -V
#$ -q UI
#$ -pe smp 2
#$ -l mem_free=2G
#$ -tc 200
#$ -j y
#$ -o results/ecms_sim/sge_logs/

module load stack/2022.2              # newest R on Argon lives in this stack
module load r/4.2.2_gcc-9.5.0         # R 4.2.2 (server runs 4.4.1: see equivalence check)

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ECMS_WGCNA_THREADS=2
SC=$1; TYPE=$2; T=$SGE_TASK_ID

if [ "$TYPE" = "ownnull" ]; then
  REP=$(( (T - 1) / 30 + 1 )); J=$(( (T - 1) % 30 + 1 ))
  OUTF=$(printf "results/ecms_sim/%s/ownnull_%04d_%03d.csv" "$SC" "$REP" "$J")
  ARGS="$SC ownnull $REP $J"
else
  OUTF=$(printf "results/ecms_sim/%s/%s_%04d.csv" "$SC" "$TYPE" "$T")
  ARGS="$SC $TYPE $T"
fi

if [ -f "$OUTF" ]; then echo "exists: $OUTF"; exit 0; fi
echo "$(date '+%F %T') job $JOB_ID task $T | host $(hostname) | $ARGS"
Rscript scripts/ecms_sim_call.R $ARGS
EXIT_STATUS=$?
echo "$(date '+%F %T') finished with status $EXIT_STATUS"
exit $EXIT_STATUS
