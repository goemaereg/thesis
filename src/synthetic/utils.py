import os
import torch


os.environ['KMP_DUPLICATE_LIB_OK']='True' # prevent weird double import error

def makedirifnot(THIS_DIR):
    if not os.path.exists(THIS_DIR):
        os.makedirs(THIS_DIR, exist_ok=True)


def manage_dir(dargs, tags=None):
    CKPT_FOLDER_DIR = dargs['checkpoint_dir']
    makedirifnot(CKPT_FOLDER_DIR)
    PROJECT_FOLDER_DIR = os.path.join(CKPT_FOLDER_DIR, dargs['project'])
    makedirifnot(PROJECT_FOLDER_DIR)

    model_name = dargs['model_name'] if 'model_name' in dargs else 'dummy'

    MODEL_DIR = os.path.join(PROJECT_FOLDER_DIR, f'{model_name}.pt')
    MODEL_INFO_DIR = os.path.join(PROJECT_FOLDER_DIR, f'{model_name}_info.json')
    LOSS_INFO_DIR = os.path.join(PROJECT_FOLDER_DIR, f'{model_name}_loss_info.json')
    HEATMAP_SAMPLE_DIR = os.path.join(PROJECT_FOLDER_DIR, f'{model_name}_heatmaps.png')

    DATA_STORE_DIR = os.path.join(PROJECT_FOLDER_DIR, 'SYNTHETIC')
    if tags:
        DATA_STORE_DIR = os.path.join(DATA_STORE_DIR, *tags)
    DATA_FOLDER_DIR = os.path.join(DATA_STORE_DIR, 'dataset')

    DATA_TRAIN_FOLDER_DIR = os.path.join(DATA_FOLDER_DIR, 'train')
    DATA_VAL_FOLDER_DIR = os.path.join(DATA_FOLDER_DIR, 'val')
    DATA_TEST_FOLDER_DIR = os.path.join(DATA_FOLDER_DIR, 'test')
    makedirifnot(DATA_TRAIN_FOLDER_DIR)
    makedirifnot(DATA_VAL_FOLDER_DIR)
    makedirifnot(DATA_TEST_FOLDER_DIR)

    METADATA_FOLDER_DIR = os.path.join(DATA_STORE_DIR, 'metadata')

    METADATA_TRAIN_FOLDER_DIR = os.path.join(METADATA_FOLDER_DIR, 'train')
    METADATA_VAL_FOLDER_DIR = os.path.join(METADATA_FOLDER_DIR, 'val')
    METADATA_TEST_FOLDER_DIR = os.path.join(METADATA_FOLDER_DIR, 'test')
    makedirifnot(METADATA_TRAIN_FOLDER_DIR)
    makedirifnot(METADATA_VAL_FOLDER_DIR)
    makedirifnot(METADATA_TEST_FOLDER_DIR)

    MASKDATA_FOLDER_DIR = os.path.join(DATA_STORE_DIR, 'maskdata')

    MASKDATA_TRAIN_FOLDER_DIR = os.path.join(MASKDATA_FOLDER_DIR, 'train')
    MASKDATA_VAL_FOLDER_DIR = os.path.join(MASKDATA_FOLDER_DIR, 'val')
    MASKDATA_TEST_FOLDER_DIR = os.path.join(MASKDATA_FOLDER_DIR, 'test')
    makedirifnot(MASKDATA_TRAIN_FOLDER_DIR)
    makedirifnot(MASKDATA_VAL_FOLDER_DIR)
    makedirifnot(MASKDATA_TEST_FOLDER_DIR)

    METRIC_AGGREGATE_DIR = os.path.join(PROJECT_FOLDER_DIR, f'{model_name}_metric_aggreagate.csv')

    DIRS = {
        'CKPT_FOLDER_DIR': CKPT_FOLDER_DIR,
        'PROJECT_FOLDER_DIR': PROJECT_FOLDER_DIR,

        'MODEL_DIR': MODEL_DIR,
        'MODEL_INFO_DIR': MODEL_INFO_DIR,
        'LOSS_INFO_DIR':LOSS_INFO_DIR,
        'HEATMAP_SAMPLE_DIR': HEATMAP_SAMPLE_DIR, 

        'DATA_TRAIN_FOLDER_DIR': DATA_TRAIN_FOLDER_DIR,
        'DATA_VAL_FOLDER_DIR': DATA_VAL_FOLDER_DIR,
        'DATA_TEST_FOLDER_DIR': DATA_TEST_FOLDER_DIR,

        'METADATA_TRAIN_FOLDER_DIR': METADATA_TRAIN_FOLDER_DIR,
        'METADATA_VAL_FOLDER_DIR': METADATA_VAL_FOLDER_DIR,
        'METADATA_TEST_FOLDER_DIR': METADATA_TEST_FOLDER_DIR,

        'MASKDATA_TRAIN_FOLDER_DIR': MASKDATA_TRAIN_FOLDER_DIR,
        'MASKDATA_VAL_FOLDER_DIR': MASKDATA_VAL_FOLDER_DIR,
        'MASKDATA_TEST_FOLDER_DIR': MASKDATA_TEST_FOLDER_DIR,

        'METRIC_AGGREGATE_DIR': METRIC_AGGREGATE_DIR
    }
    return DIRS 

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def to_batch_one_tensor(x, device=None):
    # x is a numpy array in (C,H,W)
    return torch.tensor(x).to(torch.float).unsqueeze(0).to(device=device)


