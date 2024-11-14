from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from utils.data_utilities import load_output_format_file

wav_format_datasets = ['official', 'STARSS23', 'DCASE2021', 'L3DAS22', 'synth']

class BaseDataset(Dataset):
    """ Base datset for SELD task

    """
    def __init__(self, cfg, dataset, dataset_name, rooms, dataset_type='train'):
        """
        Args:
            cfg: configurations
            dataset: dataset used
            dataset_type: 'train' | 'valid' | 'test' .
                'train' and 'valid' (optional) are only used while training. 
                Either 'valid' or 'test' is only used while infering.
        """
        super().__init__()

        self.cfg = cfg
        self.dataset_type = dataset_type
        self.label_res = dataset.label_resolution
        self.max_ov = dataset.max_ov
        self.num_classes = dataset.num_classes

        self.sample_rate = cfg['data']['sample_rate']
        self.audio_feature = cfg['data']['audio_feature']
        self.audio_type = cfg['data']['audio_type']
        self.chunklen_sec = {
            'train': cfg['data']['train_chunklen_sec'],
            'valid': cfg['data']['test_chunklen_sec'],
            'test': cfg['data']['test_chunklen_sec'],}
        self.hoplen_sec = {
            'train': cfg['data']['train_hoplen_sec'], 
            'valid': cfg['data']['test_hoplen_sec'],
            'test': cfg['data']['test_hoplen_sec'],}

        # data dir
        hdf5_dir = Path(cfg.paths.hdf5_dir)
        dataset_stage = 'eval' if rooms == ['mix'] or rooms == ['split5'] else 'dev'
        if self.audio_feature in ['logmelIV', 'logmel']:
            # Only 'logmelIV' or 'logmel' feature supports online extraction
            main_data_dir = hdf5_dir.joinpath(f'data/{self.sample_rate}fs/wav')
            self.points_per_predictions = self.sample_rate * self.label_res
        else:
            # Other features excluding logmelIV are extracted offline
            main_data_dir = hdf5_dir.joinpath(f'data/{self.sample_rate}fs/feature')
            self.data_dir = main_data_dir.joinpath(dataset_stage, self.audio_feature)
            self.points_per_predictions = int(
                self.label_res / (cfg['data']['hoplen'] / self.sample_rate))
        
        # mete dir
        label_dir = hdf5_dir.joinpath('label')
        self.track_label_dir = label_dir.joinpath('track/{}'.format(dataset_stage))
        self.accdoa_label_dir = label_dir.joinpath('accdoa/{}'.format(dataset_stage))
        self.adpit_label_dir = label_dir.joinpath('adpit/{}'.format(dataset_stage))
        
        # segments_list: data path and n_th segment
        if not (rooms == ['mix'] or rooms == ['split5']): rooms = [room+'_' for room in rooms]
        rooms.sort()
        if self.dataset_type == 'train':
            indexes_path = main_data_dir.joinpath('{}/{}_{}sChunklen_{}sHoplen_train.csv'.format(
                dataset_stage, dataset_name, self.chunklen_sec['train'], self.hoplen_sec['train']))
        elif self.dataset_type in ['valid', 'test']:
            indexes_path = main_data_dir.joinpath('{}/{}_{}sChunklen_{}sHoplen_test.csv'.format(
                dataset_stage, dataset_name, self.chunklen_sec['test'], self.hoplen_sec['test']))
        print(indexes_path)
        segments_indexes = pd.read_csv(indexes_path, header=None).values
        self.segments_list = [_segment for _segment in segments_indexes 
                            for _room in rooms if _room in _segment[0]]
        print(f'{dataset_name} {dataset_type} dataset: {len(self.segments_list)} segments')
        if dataset_name not in wav_format_datasets:
            for i in range(len(self.segments_list)):
                self.segments_list[i][0] = self.segments_list[i][0].replace('.wav', '.flac')

        if self.dataset_type in ['valid', 'test']:
            self.paths_dict = OrderedDict() # {path: num_frames}
            for segment in self.segments_list:
                self.paths_dict[segment[0]] = int(
                    np.ceil(segment[2] / self.points_per_predictions))
        if self.dataset_type == 'valid':
            # load metadata
            self.valid_gt_dcaseformat = OrderedDict() # {path: metrics_dict}
            for segment in self.segments_list:
                if segment[0] not in self.valid_gt_dcaseformat:
                    metafile = segment[0].replace('foa', 'metadata').replace('.flac', '.csv')
                    if dataset_name in wav_format_datasets: 
                        metafile = metafile.replace('.wav', '.csv')
                    if dataset_name == 'L3DAS22':
                        metafile = metafile.replace('/data_', '/metadata_')
                    self.valid_gt_dcaseformat[segment[0]] = load_output_format_file(metafile)
            # self.meta_dir = label_dir.joinpath(f'metadata/{dataset_name}')
            # for file in self.paths_dict:
            #     filename = Path(file).stem
            #     path = self.meta_dir.joinpath(str(filename) + '.csv')
            #     self.valid_gt_dcaseformat[file] = load_output_format_file(path)


    def __len__(self):
        """Get length of the dataset

        """
        return len(self.segments_list)

    def __getitem__(self, idx):
        """
        Read features from the dataset
        """
        raise NotImplementedError


