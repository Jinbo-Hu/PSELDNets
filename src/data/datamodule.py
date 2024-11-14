import numpy as np
import lightning as L
import torch
from torch.utils.data import DataLoader, ConcatDataset
from .components.sampler import UserDistributedBatchSampler
from utils.utilities import get_pylogger
from collections import OrderedDict


log = get_pylogger(__name__)


class SELDDataModule(L.LightningDataModule):

    from data.data import DatasetEINV2, DatasetACCDOA, DatasetMultiACCDOA
    UserDataset = {
        'accdoa': DatasetACCDOA,
        'einv2': DatasetEINV2,
        'multi_accdoa': DatasetMultiACCDOA,
    }

    def __init__(self, cfg, dataset, stage='fit'):
        super().__init__()
        self.cfg = cfg
        self.dataset = dataset        
        self.seed = cfg.seed

        self.paths_dict = OrderedDict()
        self.valid_gt_dcaseformat = OrderedDict()

        method = cfg.model.method

        if stage == 'fit':
            self.train_set, self.val_set = [], []

            for dataset_name, rooms in cfg.data.train_dataset.items():
                self.train_set.append(
                    self.UserDataset[method](cfg, dataset, dataset_name, rooms, 'train'))
            self.train_set = ConcatDataset(self.train_set)

            for dataset_name, rooms in cfg.data.valid_dataset.items():
                self.val_set.append(
                    self.UserDataset[method](cfg, dataset, dataset_name, rooms, 'valid'))
                self.paths_dict.update(self.val_set[-1].paths_dict)
                self.valid_gt_dcaseformat.update(self.val_set[-1].valid_gt_dcaseformat)
            # TODO: temporary implementation, need to be multi-dataloaders
            self.val_set = ConcatDataset(self.val_set)

            self.train_batch_size = cfg['model']['batch_size']
            log.info(f"Training clip number is: {len(self.train_set)}")
            log.info(f"Validation clip number is: {len(self.val_set)}")

        elif stage == 'test':
            self.test_set = []
            for dataset_name, rooms in cfg.data.test_dataset.items():
                self.test_set.append(
                    self.UserDataset[method](cfg, dataset, dataset_name, rooms, 'test'))
                self.paths_dict.update(self.test_set[-1].paths_dict)
            self.test_batch_size = cfg['model']['batch_size']
            self.test_set = ConcatDataset(self.test_set)
            log.info(f"Testing clip number is: {len(self.test_set)}")
    
    def train_dataloader(self):
        batch_sampler = UserDistributedBatchSampler(
            clip_num=len(self.train_set), 
            batch_size=self.train_batch_size,
            seed=self.seed)
        log.info(f"Number of batches per epoch is: {len(batch_sampler)}")
        
        return DataLoader(
            dataset=self.train_set,
            batch_sampler=batch_sampler,
            num_workers=self.cfg.num_workers,
            generator=torch.Generator().manual_seed(self.seed),
            pin_memory=True)
    
    def val_dataloader(self):
        rank = self.trainer.local_rank
        world_size = self.trainer.world_size
        # each gpu use different sampler (the same as DistributedSampler)
        # self.val_set.segments_list = self.val_set.segments_list[rank::world_size]

        return DataLoader(
            dataset=self.val_set,
            batch_size=self.train_batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            generator=torch.Generator().manual_seed(self.seed),
            pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_set,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            generator=torch.Generator().manual_seed(self.seed),
            pin_memory=True)