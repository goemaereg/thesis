import requests
import argparse
import json

CERT = '/Users/goemaereg/.certs/login_decrypted_ilabt_imec_be_s0175908@ua.ac.be.pem'
URI = 'https://gpulab.ilabt.imec.be/api/gpulab/v3.0'
CMD = 'jobs'

cert_file_path = CERT
key_file_path = CERT
url = f'{URI}/{CMD}'
params = {}
cert = (cert_file_path, key_file_path)
owner = {
    "userUrn": "urn:publicid:IDN+ilabt.imec.be+user+s0175908",
    "userEmail": "Geert.Goemaere@student.uantwerpen.be",
    "projectUrn": "urn:publicid:IDN+ilabt.imec.be+project+stu-weak-multi-local"
}

job = {
    "name": "pytorch",
    "description": "Used to run pytorch jobs",
    "deploymentEnvironment": "production",
    "owner": owner,
    "request": {
        "resources": {
            "cpus": 2,
            "gpus": 1,
            "cpuMemoryGb": 16,
            "gpuModel": [ "V100" ]
        },
        "docker": {
            "environment": {
                "EXPERIMENT_NAME": "",
                "RUN_NAME": "",
                "CONFIG_FILE": "",
                "PARAMETERS": ""
            },
            "storage": [{
                "hostPath": "/project_antwerp",
                "containerPath": "/project"
            }],
            "image": "pytorch/pytorch",
            "command": "/project/thesis/start_job.sh"
        },
        "scheduling": {
            "interactive": False,
            "restartable": False,
            "minDuration": "4 days",
            "maxDuration": "1 week"
        }
    }
}


def dict_merge(dct, merge_dct):
    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict) and isinstance(merge_dct[k], dict)):  #noqa
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]
    return dct

def main(job):
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment_name', type=str, required=True)
    parser.add_argument('-r', '--run_name', type=str, required=True)
    parser.add_argument('-c', '--config_file', type=str, required=True)
    parser.add_argument('-p', '--parameters', type=str, required=False, help='overrides config_file params. Form: ^A=a^B=b')
    parser.add_argument('-j', '--job', type=str, required=False, default='jobs/job_default.json')
    args = parser.parse_args()
    job_overlay = {}
    with open(args.job, 'r') as fp:
        job_overlay = json.load(fp)
    env = dict(EXPERIMENT_NAME=args.experiment_name,
               RUN_NAME=args.run_name,
               CONFIG_FILE=args.config_file)
    if args.parameters:
        env |= dict(PARAMETERS=args.parameters)
    job = dict_merge(job, job_overlay)
    job['request']['docker']['environment'] = env
    r = requests.post(url, json=job, cert=cert)
    print(r.status_code)
    print(r.text)


if __name__ == '__main__':
    main(job)

