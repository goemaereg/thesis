mlflow run -e main --experiment-name $EXPERIMENT_NAME --run-name $RUN_NAME --env-manager local -P config=$CONFIG_FILE -P override_cache=true .
