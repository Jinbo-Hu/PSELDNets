from utils.datasets import *
import utils.feature as feature


dataset_dict = {
    'STARSS23': STARSS23,
    'synth': Synthesis,
    'DCASE2021': DCASE2021TASK3,
    'L3DAS22': L3DAS22,
}

# Datasets
def get_dataset(dataset_name, cfg):
    root_dir = cfg.paths.dataset_dir
    if 'Recording' in dataset_name:
        dataset_name = 'recording'
    elif dataset_name not in dataset_dict.keys():
        dataset_name = 'synth'
    dataset = dataset_dict[dataset_name](root_dir, cfg)
    print('\nDataset {} is being developed......\n'.format(dataset_name))
    return dataset


def get_afextractor(cfg):
    """ Get audio feature extractor."""
    if cfg['data']['audio_feature'] == 'logmelIV':
        afextractor = feature.LogmelIV_Extractor(cfg)
    elif cfg['data']['audio_feature'] == 'logmel':
        afextractor = feature.Logmel_Extractor(cfg)
    else:
        afextractor = None
    return afextractor