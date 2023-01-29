# Computer Science Master Thesis

## Training
### System environment
mlflow run -e main --experiment-name wsol_test --run-name test_syn_o2t --env-manager local -P config=files/config/config_train_vgg16_cam_synthetic_d2t.json -P override_cache=true -P epochs=20 .

### Virtual environment
mlflow run -e main --experiment-name wsol_test --run-name test_syn_o2t -P config=files/config/config_train_vgg16_cam_synthetic_d2t.json -P override_cache=true -P epochs=20 .

### Docker
image: pytorch/pytorch

Then, ssh into docker container:
cd /project/thesis
source venv/bin/activate
apt-get install libgl1
apt-get install libgtk2.0-dev
