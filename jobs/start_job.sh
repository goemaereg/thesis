#!/bin/bash

halt_handler() {
   echo "saw SIGUSR1 @ $(date)"
   # TODO: if needed, make the job save its work and stop
   exit 123
}
trap halt_handler SIGUSR1;

exec_job() {
   cd /project/thesis
   apt-get update
   apt-get install -y vim rsync git
   apt-get install -y libgl1
   apt-get install -y libgtk2.0-dev
   source venv/bin/activate
   mlflow run -e main --experiment-name $EXPERIMENT_NAME --run-name $RUN_NAME --env-manager local -P config=$CONFIG_FILE -P override_cache=true .
}

echo "Job started at $(date)"
echo "GPULAB_JOB_ID=${GPULAB_JOB_ID}"  # different for each restart
echo "GPULAB_RESTART_INITIAL_JOB_ID=${GPULAB_RESTART_INITIAL_JOB_ID}"  # always the same, for original and restarts
echo "GPULAB_RESTART_COUNT=${GPULAB_RESTART_COUNT}"

if [ $GPULAB_RESTART_COUNT -lt 1 ]
then
   exec_job
else
   # TODO: restart_my_job
   echo "Job restart logic is not implemented"
fi

