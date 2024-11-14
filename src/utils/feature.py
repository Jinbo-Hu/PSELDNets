import torch
import torch.nn as nn
import librosa
import numpy as np
import torchaudio
import math

eps = torch.finfo(torch.float32).eps
window_fn_dict = {
    'hann': torch.hann_window,
    'hamming': torch.hamming_window,
    'blackman': torch.blackman_window,
    'bartlett': torch.bartlett_window,
}

def nCr(n, r):
    return math.factorial(n) // math.factorial(r) // math.factorial(n-r)


class LogmelIV_Extractor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        data = cfg['data']
        assert data['window'] in window_fn_dict.keys(), \
            "window must be in {}, but got {}".format(window_fn_dict.keys(), data['window'])
        
        self.stft_extractor = torchaudio.transforms.Spectrogram(
            n_fft=data['nfft'], hop_length=data['hoplen'], 
            win_length=data['nfft'], window_fn=window_fn_dict[data['window']], 
            power=None,)
        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=data['n_mels'], sample_rate=data['sample_rate'], norm='slaney',
            f_min=20, f_max=data['sample_rate']/2, n_stft=data['nfft']//2+1,)
        self.amp2db = torchaudio.transforms.AmplitudeToDB(
            stype='power', top_db=None, )
        self.intensity_vector = intensityvector
    
    def forward(self, x):
        """
        input: 
            (batch_size, channels=4, data_length)
        output: 
            (batch_size, channels, time_steps, freq_bins) freq_bins->mel_bins
        """
        if x.ndim != 3:
            raise ValueError("x shape must be (batch_size, num_channels, data_length)\n \
                            Now it is {}".format(x.shape))
        x = self.stft_extractor(x)
        mel = self.mel_scale(torch.abs(x)**2)
        logmel = self.amp2db(mel).transpose(-1, -2)
        intensity_vector = self.intensity_vector(
            [x.real.transpose(-1, -2), x.imag.transpose(-1, -2)], 
            self.mel_scale.fb)
        out = torch.cat((logmel, intensity_vector), dim=1)
        return out


class Logmel_Extractor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        data = cfg['data']
        assert data['window'] in window_fn_dict.keys(), \
            "window must be in {}, but got {}".format(window_fn_dict.keys(), data['window'])
        
        self.stft_extractor = torchaudio.transforms.Spectrogram(
            n_fft=data['nfft'], hop_length=data['hoplen'], 
            win_length=data['nfft'], window_fn=window_fn_dict[data['window']], 
            power=None,)
        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=data['n_mels'], sample_rate=data['sample_rate'], norm='slaney',
            f_min=20, f_max=data['sample_rate']/2, n_stft=data['nfft']//2+1,)
        self.amp2db = torchaudio.transforms.AmplitudeToDB(
            stype='power', top_db=None, )
    
    def forward(self, x):
        """
        input: 
            (batch_size, channels=4, data_length)
        output: 
            (batch_size, channels, time_steps, freq_bins) freq_bins->mel_bins
        """
        if x.ndim != 3:
            raise ValueError("x shape must be (batch_size, num_channels, data_length)\n \
                            Now it is {}".format(x.shape))
        x = self.stft_extractor(x)
        mel = self.mel_scale(torch.abs(x)**2)
        logmel = self.amp2db(mel).transpose(-1, -2)
        out = logmel
        return out

def intensityvector(input, melW):
    """Calculate intensity vector. Input is four channel stft of the signals.
    input: (stft_real, stft_imag)
        stft_real: (batch_size, 4, time_steps, freq_bins)
        stft_imag: (batch_size, 4, time_steps, freq_bins)
    out:
        intenVec: (batch_size, 3, time_steps, freq_bins)
    """
    sig_real, sig_imag = input[0], input[1]
    Pref_real, Pref_imag = sig_real[:,0,...], sig_imag[:,0,...]
    Px_real, Px_imag = sig_real[:,1,...], sig_imag[:,1,...]
    Py_real, Py_imag = sig_real[:,2,...], sig_imag[:,2,...]
    Pz_real, Pz_imag = sig_real[:,3,...], sig_imag[:,3,...]

    IVx = Pref_real * Px_real + Pref_imag * Px_imag
    IVy = Pref_real * Py_real + Pref_imag * Py_imag
    IVz = Pref_real * Pz_real + Pref_imag * Pz_imag
    normal = torch.sqrt(IVx**2 + IVy**2 + IVz**2) + eps

    IVx_mel = torch.matmul(IVx / normal, melW)
    IVy_mel = torch.matmul(IVy / normal, melW)
    IVz_mel = torch.matmul(IVz / normal, melW)
    intenVec = torch.stack([IVx_mel, IVy_mel, IVz_mel], dim=1)

    return intenVec

class Features_Extractor_MIC():
    def __init__(self, cfg):
        self.fs = cfg['data']['sample_rate']
        self.n_fft = cfg['data']['nfft']
        self.n_mels = cfg['data']['n_mels']
        self.hoplen = cfg['data']['hoplen']
        self.window = cfg['data']['window']
        self.mel_bank = librosa.filters.mel(sr=self.fs, n_fft=self.n_fft, n_mels=self.n_mels).T
        if cfg['data']['audio_feature'] == 'salsalite':
            # Initialize the spatial feature constants
            c = 343
            self.fmin_doa = cfg['data']['salsalite']['fmin_doa']
            self.fmax_doa = cfg['data']['salsalite']['fmax_doa']
            self.fmax_spectra = cfg['data']['salsalite']['fmax_spectra']

            self.lower_bin = np.int(np.floor(self.fmin_doa * self.n_fft / np.float(self.fs)))
            self.lower_bin = np.max((self.lower_bin, 1))
            self.upper_bin = np.int(np.floor(np.min((self.fmax_doa, self.fs//2)) * self.n_fft / np.float(self.fs)))
            self.cutoff_bin = np.int(np.floor(self.fmax_spectra * self.n_fft / np.float(self.fs)))
            assert self.upper_bin <= self.cutoff_bin, 'Upper bin for doa feature is higher than cutoff bin for spectrogram {}!'
            
            # Normalization factor for salsalite
            self.delta = 2 * np.pi * self.fs / (self.n_fft * c)
            self.freq_vector = np.arange(self.n_fft // 2 + 1)
            self.freq_vector[0] = 1
            self.freq_vector = self.freq_vector[None, :, None]

    def _spectrogram(self, audio_input, _nb_frames):
        _nb_ch = audio_input.shape[1]
        spectra = []
        for ch_cnt in range(_nb_ch):
            stft_ch = librosa.core.stft(np.asfortranarray(audio_input[:, ch_cnt]), n_fft=self.n_fft, hop_length=self.hoplen,
                                        win_length=self.n_fft, window=self.window)
            spectra.append(stft_ch[:, :_nb_frames])
        return np.array(spectra).T

    def _get_logmel_spectrogram(self, linear_spectra):
        logmel_feat = np.zeros((linear_spectra.shape[0], self.n_mels, linear_spectra.shape[-1]))
        for ch_cnt in range(linear_spectra.shape[-1]):
            mag_spectra = np.abs(linear_spectra[:, :, ch_cnt])**2
            mel_spectra = np.dot(mag_spectra, self.mel_bank)
            logmel_spectra = librosa.power_to_db(mel_spectra)
            logmel_feat[:, :, ch_cnt] = logmel_spectra
        return logmel_feat

    def _get_gcc(self, linear_spectra):
        gcc_channels = nCr(linear_spectra.shape[-1], 2)
        gcc_feat = np.zeros((linear_spectra.shape[0], self.n_mels, gcc_channels))
        cnt = 0
        for m in range(linear_spectra.shape[-1]):
            for n in range(m+1, linear_spectra.shape[-1]):
                R = np.conj(linear_spectra[:, :, m]) * linear_spectra[:, :, n]
                cc = np.fft.irfft(np.exp(1.j*np.angle(R)))
                cc = np.concatenate((cc[:, -self.n_mels//2:], cc[:, :self.n_mels//2]), axis=-1)
                gcc_feat[:, :, cnt] = cc
                cnt += 1
        return gcc_feat

    def _get_salsalite(self, linear_spectra):
        # Adapted from the official SALSA repo- https://github.com/thomeou/SALSA
        # spatial features
        phase_vector = np.angle(linear_spectra[:, :, 1:] * np.conj(linear_spectra[:, :, 0, None]))
        phase_vector = phase_vector / (self.delta * self.freq_vector)
        phase_vector = phase_vector[:, self.lower_bin:self.cutoff_bin, :]
        phase_vector[:, self.upper_bin:, :] = 0
        phase_vector = phase_vector.transpose((2, 0, 1))

        # spectral features
        linear_spectra = np.abs(linear_spectra)**2
        for ch_cnt in range(linear_spectra.shape[-1]):
            linear_spectra[:, :, ch_cnt] = librosa.power_to_db(linear_spectra[:, :, ch_cnt], ref=1.0, amin=1e-10, top_db=None)
        linear_spectra = linear_spectra[:, self.lower_bin:self.cutoff_bin, :]
        linear_spectra = linear_spectra.transpose((2, 0, 1))
        
        return np.concatenate((linear_spectra, phase_vector), axis=0) 