from pathlib import Path

class BaseDataset:
    """ Base Dataset class
    """    
    def __init__(self):
    
        self.label_resolution = 0.1

        self.label_dic = dict()
        self.max_ov = None
        self.num_classes = len(self.label_dic)
        self.root_dir = None
        self.dataset_dir = dict()
        self.dataset_dir['dev'] = {
            'foa': None,
            'mic': None,
            'meta': None,
        }
        self.dataset_dir['eval'] = {
            'foa': None,
            'mic': None,
            'meta': None,
        }


class Synthesis(BaseDataset):
    ''' Simulate dataset
    '''
    def __init__(self, root_dir, cfg):
        super().__init__()
        
        # dataset = cfg.data.train_dataset[0]
        # dataset = list(cfg.data.train_dataset.keys())[0]
        dataset = cfg.get('dataset', list(cfg.data.train_dataset.keys())[0])
        
        self.max_ov = 3
        self.root_dir = Path(root_dir).joinpath(f'{dataset}')

        if dataset != 'official':
            cls_indices_path = self.root_dir.parent.joinpath('cls_indices_train.tsv')
            with open(cls_indices_path, 'r') as f:
                for idx, line in enumerate(f.readlines()):
                    line = line.strip().split('\t')
                    self.label_dic[line[2]] = idx
        else:
            self.label_dic = {'Female speech, woman speaking': 0,
                            'Male speech, man speaking': 1,
                            'Clapping': 2,
                            'Telephone': 3,
                            'Laughter': 4,
                            'Domestic sounds': 5,
                            'Walk, footsteps': 6,
                            'Door, open or close': 7,
                            'Music': 8,
                            'Musical instrument': 9,
                            'Water tap, faucet': 10,
                            'Bell': 11,
                            'Knock': 12 }
        self.num_classes = len(self.label_dic)

        self.dataset_dir['dev'] = {
            'foa': self.root_dir.joinpath('foa'),
            'mic': self.root_dir.joinpath('mic'),
            'meta': self.root_dir.joinpath('metadata'),
        }
        self.dataset_dir['eval'] = {
            'foa': self.root_dir.joinpath('foa'),
            'mic': self.root_dir.joinpath('mic'),
            'meta': self.root_dir.joinpath('metadata'),
        }


class DCASE2021TASK3(BaseDataset):
    ''' dcase2021task3 dataset 
    '''
    def __init__(self, root_dir, cfg):
        super().__init__()
        self.label_dic = {'alarm': 0,
                        'crying baby': 1,
                        'crash': 2,
                        'barking dog': 3,
                        'female scream': 4,
                        'female speech': 5,
                        'footsteps': 6,
                        'knocking on door': 7,
                        'male scream': 8,
                        'male speech': 9,
                        'ringing phone': 10,
                        'piano': 11,}

        self.max_ov = 3 # max overlap
        self.num_classes = len(self.label_dic)
        self.root_dir = Path(root_dir).joinpath('DCASE2021')

        self.dataset_dir['dev']['foa'] = self.root_dir / 'foa_dev'
        self.dataset_dir['dev']['mic'] = self.root_dir / 'mic_dev'
        self.dataset_dir['dev']['meta'] = self.root_dir / 'metadata_dev'
        self.dataset_dir['eval'] = {
            'foa': self.root_dir.joinpath('foa_eval'),
            'mic': self.root_dir.joinpath('mic_eval'),
            'meta': self.root_dir.joinpath('metadata_eval'),
        }


class STARSS23(BaseDataset):
    ''' STARSS23 dataset for DCASE2023-Task3
    '''
    def __init__(self, root_dir, cfg):
        super().__init__()
        self.label_dic = {'Female speech, woman speaking': 0,
                            'Male speech, man speaking': 1,
                            'Clapping': 2,
                            'Telephone': 3,
                            'Laughter': 4,
                            'Domestic sounds': 5,
                            'Walk, footsteps': 6,
                            'Door, open or close': 7,
                            'Music': 8,
                            'Musical instrument': 9,
                            'Water tap, faucet': 10,
                            'Bell': 11,
                            'Knock': 12 }

        self.max_ov = 3 # max overlap
        self.num_classes = len(self.label_dic)
        self.root_dir = Path(root_dir).joinpath('STARSS23')

        self.dataset_dir['dev']['foa'] = self.root_dir / 'foa_dev'
        self.dataset_dir['dev']['mic'] = self.root_dir / 'mic_dev'
        self.dataset_dir['dev']['meta'] = self.root_dir / 'metadata_dev'
        self.dataset_dir['eval'] = {
            'foa': self.root_dir / 'foa_eval',
            'mic': self.root_dir / 'mic_eval',
            'meta': None,
        }

class L3DAS22(BaseDataset):
    ''' L3DAS22 dataset

    '''
    def __init__(self, root_dir, cfg):
        super().__init__()
        self.root_dir = Path(root_dir).joinpath('L3DAS22')
        self.dataset_dir = dict()
        self.clip_length = 30
        self.dataset_dir['dev'] = {
            'foa': self.root_dir.joinpath('data_train'),
            'mic': None,
            'label': self.root_dir.joinpath('labels_train'),
            'meta': self.root_dir.joinpath('metadata_train')
        }
        self.dataset_dir['eval'] = {
            'foa': self.root_dir.joinpath('data_test'),
            'mic': None,
            'label': self.root_dir.joinpath('labels_test'), 
            'meta': self.root_dir.joinpath('metadata_test')
        }
        self.label_set = ['Computer_keyboard', 'Drawer_open_or_close', 'Cupboard_open_or_close',
                          'Finger_snapping', 'Keys_jangling', 'Knock', 'Laughter', 'Scissors', 'Telephone',
                          'Writing', 'Chink_and_clink', 'Printer', 'Female_speech_and_woman_speaking', 
                          'Male_speech_and_man_speaking']
        self.label_dic = {'Chink_and_clink':0,
                          'Computer_keyboard':1,
                          'Cupboard_open_or_close':2,
                          'Drawer_open_or_close':3,
                          'Female_speech_and_woman_speaking':4,
                          'Finger_snapping':5,
                          'Keys_jangling':6,
                          'Knock':7,
                          'Laughter':8,
                          'Male_speech_and_man_speaking':9,
                          'Printer':10,
                          'Scissors':11,
                          'Telephone':12,
                          'Writing':13}

        self.max_ov = 3 # max overlap
        self.num_classes = len(self.label_set)
        