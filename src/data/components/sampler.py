import numpy as np
from torch.utils.data import Sampler
import torch.distributed as dist

class UserDistributedBatchSampler(Sampler):
    """User defined distributed batch sampler. Only for train set.

    """
    def __init__(self, clip_num, batch_size=1, seed=2023, data_indices=None,
                 shuffle=True, last_batch_supplement=True):
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        self.clip_num = clip_num
        self.batch_size = batch_size * self.num_replicas

        if data_indices is None:
            self.indices = np.arange(clip_num)
        else:
            self.indices = data_indices
            self.clip_num = len(data_indices)
        
        self.pointer = 0
        self.shuffle = shuffle
        if self.shuffle:
            self.random_state = np.random.RandomState(seed)
            self.random_state.shuffle(self.indices)
        
        if last_batch_supplement:
            padding_size = self.batch_size - self.clip_num % self.batch_size
            self.indices = np.append(self.indices, self.indices[:padding_size])
            self.clip_num = self.clip_num + padding_size
    
    def __iter__(self):
        """
        Return: 
            batch_indices (int): indices of batch
        """   
        while True:
            if self.pointer >= self.clip_num:
                self.pointer = 0
                if self.shuffle:
                    self.random_state.shuffle(self.indices)

            batch_indices = self.indices[self.pointer+self.rank: self.pointer+self.batch_size: self.num_replicas]
            self.pointer += self.batch_size
            yield batch_indices

    def __len__(self):
        return np.ceil(self.clip_num / self.batch_size).astype(int)


class UserBatchSampler(Sampler):
    """User defined batch sampler. Only for train set.

    """
    def __init__(self, clip_num, batch_size=1, seed=2023, data_indices=None,
                 shuffle=True, last_batch_supplement=True):
        self.clip_num = clip_num
        self.batch_size = batch_size

        if data_indices is None:
            self.indices = np.arange(clip_num)
        else:
            self.indices = data_indices
            self.clip_num = len(data_indices)
        
        self.pointer = 0
        self.shuffle = shuffle
        if self.shuffle:
            self.random_state = np.random.RandomState(seed)
            self.random_state.shuffle(self.indices)
        
        if last_batch_supplement:
            padding_size = self.batch_size - self.clip_num % self.batch_size
            self.indices = np.append(self.indices, self.indices[:padding_size])
            self.clip_num = self.clip_num + padding_size
    
    def __iter__(self):
        """
        Return: 
            batch_indices (int): indices of batch
        """   
        while True:
            if self.pointer >= self.clip_num:
                self.pointer = 0
                if self.shuffle:
                    self.random_state.shuffle(self.indices)

            batch_indices = self.indices[self.pointer: self.pointer+self.batch_size]
            self.pointer += self.batch_size
            yield batch_indices

    def __len__(self):
        return np.ceil(self.clip_num / self.batch_size).astype(int)
