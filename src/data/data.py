from pathlib import Path
import h5py
import numpy as np
from .components.data import BaseDataset
import soundfile as sf

def load_audio(path, index_begin, index_end):
    try:
        x = sf.read(path, dtype='float32', start=index_begin, stop=index_end)[0].T
    except:
        # print(path, index_begin, index_end)
        # may encounter error when reading partial segments of audio file in some cases
        x = sf.read(path, dtype='float32')[0].T
        x = x[:, index_begin:index_end]
    return x

def generate_spatial_samples(audio, method, **kwargs):
    # NOTE: only support single-source target

    if audio.ndim == 2:
        audio = audio[0]

    azi = np.random.randint(-180, 180)
    ele = np.random.randint(-90, 90)
    w = audio
    x = np.cos(np.deg2rad(azi)) * np.cos(np.deg2rad(ele))
    y = np.sin(np.deg2rad(azi)) * np.cos(np.deg2rad(ele))
    z = np.sin(np.deg2rad(ele))
    audio = np.stack((w, y * audio, z * audio, x * audio), axis=0)

    if method == 'einv2':
        sed_label, doa_label = kwargs['sed_label'], kwargs['doa_label']
        assert sed_label.sum(axis=-2).max() <= 1
        doa_label = np.zeros_like(doa_label)
        doa_label[..., 0, 0] = sed_label.sum(axis=(-1, -2)) * x
        doa_label[..., 0, 1] = sed_label.sum(axis=(-1, -2)) * y
        doa_label[..., 0, 2] = sed_label.sum(axis=(-1, -2)) * z
        return audio, sed_label, doa_label
    elif method == 'accdoa':
        accdoa_label = kwargs['accdoa_label']
        num_classes = accdoa_label.shape[-1] // 4
        se_label = accdoa_label[:, :num_classes]
        assert se_label.sum(axis=-1).max() <= 1
        accdoa_label = np.zeros_like(accdoa_label)
        accdoa_label[..., num_classes:num_classes*2] = x * se_label
        accdoa_label[..., num_classes*2:num_classes*3] = y * se_label
        accdoa_label[..., num_classes*3:] = z * se_label
        return audio, accdoa_label
    elif method == 'multi_accdoa':
        adpit_label = kwargs['adpit_label']
        num_classes = adpit_label.shape[-1]
        se_label = adpit_label[:, :, 0, :]
        assert se_label.sum(axis=(-1, -2)).max() <= 1
        adpit_label = np.zeros_like(adpit_label)
        adpit_label[:, :, 0, :] = se_label
        adpit_label[:, :, 1, :] = x * se_label
        adpit_label[:, :, 2, :] = y * se_label
        adpit_label[:, :, 3, :] = z * se_label
        return audio, adpit_label


class DatasetACCDOA(BaseDataset):
    
    def __getitem__(self, idx):
        """ Datset for the ACCDOA method of SELD task

        """
        clip_indexes = self.segments_list[idx]
        path, segments = clip_indexes[0], clip_indexes[1:]
        fn = Path(path).stem
        index_begin = segments[0]
        index_end = segments[1]
        pad_width_before = segments[2]
        pad_width_after = segments[3]
        if self.audio_feature in ['logmelIV']:
            data_path = path
            x = load_audio(data_path, index_begin, index_end)
            pad_width = ((0, 0), (pad_width_before, pad_width_after))
            dataset = path.split('/')[-3]
        else:
            data_path = self.data_dir.joinpath(path)
            with h5py.File(data_path, 'r') as hf:
                x = hf['feature'][:, index_begin: index_end] 
            pad_width = ((0, 0), (pad_width_before, pad_width_after), (0, 0))
            dataset = path.split('/')[-2]
        x = np.pad(x, pad_width, mode='constant')

        if self.dataset_type != 'test':
            meta_path = self.accdoa_label_dir.joinpath('{}.h5'.format(dataset))
            index_begin_label = int(index_begin / self.points_per_predictions)
            index_end_label = int(index_end / self.points_per_predictions)
            with h5py.File(meta_path, 'r') as hf:
                # accdoa_label = hf[f'{fn}/accdoa'][index_begin_label: index_end_label, ...]
                se_label = hf[f'{fn}/accdoa/se'][index_begin_label: index_end_label, ...].astype(np.float32)
                azi_label = hf[f'{fn}/accdoa/azi'][index_begin_label: index_end_label, ...].astype(np.float32)
                ele_label = hf[f'{fn}/accdoa/ele'][index_begin_label: index_end_label, ...].astype(np.float32)
                lx = np.cos(np.deg2rad(azi_label)) * np.cos(np.deg2rad(ele_label)) * se_label
                ly = np.sin(np.deg2rad(azi_label)) * np.cos(np.deg2rad(ele_label)) * se_label
                lz = np.sin(np.deg2rad(ele_label)) * se_label
                del azi_label, ele_label
                accdoa_label = np.concatenate((se_label, lx, ly, lz), axis=1, dtype=np.float32)
            pad_width_after_label = int(
                self.chunklen_sec[self.dataset_type] / self.label_res - accdoa_label.shape[0])
            if pad_width_after_label != 0:
                accdoa_label_new = np.zeros(
                    (pad_width_after_label, 4*self.num_classes), 
                    dtype=np.float32)
                accdoa_label = np.concatenate((accdoa_label, accdoa_label_new), axis=0)
        if self.dataset_type == 'train' and self.cfg.adapt.method == 'mono_adapter':
            x, accdoa_label = generate_spatial_samples(
                x, method='accdoa', accdoa_label=accdoa_label)
        if self.dataset_type != 'test':
            ov = str(max(np.sum(accdoa_label[:, :self.num_classes], axis=1).max().astype(int), 1))

            sample = {
                'filename': path,
                'data': x,
                'accdoa_label': accdoa_label[:, self.num_classes:],
                'ov': ov
            }
        else:
            sample = {
                'filename': path,
                'data': x
            }
          
        return sample    
    

class DatasetEINV2(BaseDataset):

    def __getitem__(self, idx):
        """ Datset for the EINV2 method of SELD task

        """
        clip_indexes = self.segments_list[idx]
        path, segments = clip_indexes[0], clip_indexes[1:]
        fn = Path(path).stem
        index_begin = segments[0]
        index_end = segments[1]
        pad_width_before = segments[2]
        pad_width_after = segments[3]
        if self.audio_feature in ['logmelIV']:
            data_path = path
            x = load_audio(data_path, index_begin, index_end)
            pad_width = ((0, 0), (pad_width_before, pad_width_after))
            dataset = path.split('/')[-3]
        else:
            data_path = self.data_dir.joinpath(path)
            with h5py.File(data_path, 'r') as hf:
                x = hf['feature'][:, index_begin: index_end] 
            pad_width = ((0, 0), (pad_width_before, pad_width_after), (0, 0))
            dataset = path.split('/')[-2]
        x = np.pad(x, pad_width, mode='constant')
        if self.dataset_type != 'test':
            meta_path = self.track_label_dir.joinpath('{}.h5'.format(dataset))
            index_begin_label = int(index_begin / self.points_per_predictions)
            index_end_label = int(index_end / self.points_per_predictions)
            with h5py.File(meta_path, 'r') as hf:
                sed_label = hf[f'{fn}/sed_label'][index_begin_label: index_end_label, :self.max_ov]
                doa_label = hf[f'{fn}/doa_label'][index_begin_label: index_end_label, :self.max_ov]
            pad_width_after_label = int(
                self.chunklen_sec[self.dataset_type] / self.label_res - sed_label.shape[0])
            if pad_width_after_label != 0:
                sed_label_new = np.zeros((pad_width_after_label, self.max_ov, self.num_classes))
                sed_label = np.concatenate((sed_label, sed_label_new), axis=0)
                doa_label_new = np.zeros((pad_width_after_label, self.max_ov, 3))
                doa_label = np.concatenate((doa_label, doa_label_new), axis=0)
        if self.dataset_type == 'train' and self.cfg.adapt.method == 'mono_adapter':
            x, sed_label, doa_label = generate_spatial_samples(
                x, method='einv2', sed_label=sed_label, doa_label=doa_label)
        if self.dataset_type != 'test':
            ov = str(max(np.sum(sed_label, axis=(1,2)).max().astype(int), 1))
            sample = {
                'filename': path,
                'data': x,
                'sed_label': sed_label.astype(np.float32),
                'doa_label': doa_label.astype(np.float32),
                'ov': ov
            }
        else:
            sample = {
                'filename': path,
                'data': x
            }
          
        return sample    


class DatasetMultiACCDOA(BaseDataset):

    def __getitem__(self, idx):
        """
        Read features from the dataset
        """
        clip_indexes = self.segments_list[idx]
        path, segments = clip_indexes[0], clip_indexes[1:]
        fn = Path(path).stem
        index_begin = segments[0]
        index_end = segments[1]
        pad_width_before = segments[2]
        pad_width_after = segments[3]
        if self.audio_feature in ['logmelIV']:
            data_path = path
            x = load_audio(data_path, index_begin, index_end)
            pad_width = ((0, 0), (pad_width_before, pad_width_after))
            dataset = path.split('/')[-3]
        else:
            data_path = self.data_dir.joinpath(path)
            with h5py.File(data_path, 'r') as hf:
                x = hf['feature'][:, index_begin: index_end] 
            pad_width = ((0, 0), (pad_width_before, pad_width_after), (0, 0))
            dataset = path.split('/')[-2]
        x = np.pad(x, pad_width, mode='constant')

        if 'test' not in self.dataset_type:
            meta_path = self.adpit_label_dir.joinpath('{}.h5'.format(dataset))
            index_begin_label = int(index_begin / self.points_per_predictions)
            index_end_label = int(index_end / self.points_per_predictions)
            with h5py.File(meta_path, 'r') as hf:
                # adpit_label = hf[f'{fn}/adpit'][index_begin_label: index_end_label, ...]
                se_label = hf[f'{fn}/adpit/se'][index_begin_label: index_end_label, ...].astype(np.float32)
                azi_label = hf[f'{fn}/adpit/azi'][index_begin_label: index_end_label, ...].astype(np.float32)
                ele_label = hf[f'{fn}/adpit/ele'][index_begin_label: index_end_label, ...].astype(np.float32)
                lx = np.cos(np.deg2rad(azi_label)) * np.cos(np.deg2rad(ele_label)) * se_label
                ly = np.sin(np.deg2rad(azi_label)) * np.cos(np.deg2rad(ele_label)) * se_label
                lz = np.sin(np.deg2rad(ele_label)) * se_label
                del azi_label, ele_label
                adpit_label = np.stack((se_label, lx, ly, lz), axis=2, dtype=np.float32)
            pad_width_after_label = int(
                self.chunklen_sec[self.dataset_type] / self.label_res - adpit_label.shape[0])
            if pad_width_after_label != 0:
                adpit_label_new = np.zeros((pad_width_after_label, 6, 4, self.num_classes), dtype=np.float32)
                adpit_label = np.concatenate((adpit_label, adpit_label_new), axis=0)
        if self.dataset_type == 'train' and self.cfg.adapt.method == 'mono_adapter':
            x, adpit_label = generate_spatial_samples(
                x, method='multi_accdoa', adpit_label=adpit_label)
        if 'test' not in self.dataset_type:
            ov = str(max(np.sum(adpit_label[:, :, 0, :], axis=(1,2)).max().astype(int), 1))

            sample = {
                'filename': path,
                'data': x,
                'adpit_label': adpit_label,
                'ov': ov
            }
        else:
            sample = {
                'filename': path,
                'data': x
            }
          
        return sample    