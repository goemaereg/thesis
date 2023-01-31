cd /project/thesis
source venv/bin/activate
apt-get update
apt-get install -y vim
apt-get install -y rsync
apt-get install -y libgl1
apt-get install -y libgtk2.0-dev
mlflow run -e main --experiment-name $EXPERIMENT_NAME --run-name $RUN_NAME --env-manager local -P config=$CONFIG_FILE -P override_cache=true .

