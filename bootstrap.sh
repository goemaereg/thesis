cd /project/thesis
source venv/bin/activate
apt-get install libgl1
apt-get install libgtk2.0-dev
mlflow run -e main --experiment-name $EXPERIMENT_NAME --run-name $RUN_NAME --env-manager local -P config=$CONFIG_FILE -P override_cache=true .

