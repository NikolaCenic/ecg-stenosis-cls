import torch
import random
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np

from scipy import sparse
from scipy.sparse.linalg import spsolve


class ECGNormalisation(object):
    """
    Time series normalisation.
    """

    def __init__(self, mode="group_wise", groups=[3, 6, 12]) -> None:
        self.mode = mode
        self.groups = groups

    def __call__(self, sample) -> np.array:
        sample_dtype = sample.dtype

        if self.mode == "sample_wise":
            mean = np.mean(sample)
            var = np.var(sample)

        elif self.mode == "channel_wise":
            mean = np.mean(sample, axis=-1, keepdims=True)
            var = np.var(sample, axis=-1, keepdims=True)

        elif self.mode == "group_wise":
            mean = []
            var = []

            lower_bound = 0
            for idx in self.groups:
                mean_group = np.mean(
                    sample[lower_bound:idx], axis=(0, 1), keepdims=True
                )
                mean_group = np.repeat(
                    mean_group, repeats=int(idx - lower_bound), axis=0
                )
                var_group = np.var(sample[lower_bound:idx], axis=(0, 1), keepdims=True)
                var_group = np.repeat(var_group, repeats=int(idx - lower_bound), axis=0)
                lower_bound = idx

                mean.extend(mean_group)
                var.extend(var_group)

            mean = np.array(mean, dtype=sample_dtype)
            var = np.array(var, dtype=sample_dtype)

        normalised_sample = (sample - mean) / (var + 1.0e-12) ** 0.5

        return normalised_sample


def baseline_als(y, lam=1e8, p=1e-2, niter=10):
    """
    Asymmetric Least Squares Smoothing, i.e. asymmetric weighting of deviations to correct a baseline
    while retaining the signal peak information.
    Refernce: Paul H. C. Eilers and Hans F.M. Boelens, Baseline Correction with Asymmetric Least Squares Smoothing (2005).
    """
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w)
        Z = W + D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def process_ecg(sample):
    # remove nan

    sample = np.nan_to_num(sample)

    # clamp
    sample_std = sample.std()
    sample = np.clip(sample, a_min=-4 * sample_std, a_max=4 * sample_std)

    # remove baseline wander
    baselines = np.zeros_like(sample)
    for lead in range(sample.shape[0]):
        baselines[lead] = baseline_als(sample[lead], lam=1e7, p=0.3, niter=5)
    sample = sample - baselines

    # normalise
    transform = ECGNormalisation(mode="group_wise", groups=[3, 6, 12])
    sample = transform(sample)

    return sample


class ECGTransforms:
    def __init__(self, augmentation, setting):
        self.setting = setting
        if augmentation is None:
            self.do_augmentation = False
        else:
            self.config = augmentation.config
            self.do_augmentation = augmentation.do and setting.lower() == "train"

    @staticmethod
    def sample_ecg(x, sr, ecg_length_seconds, setting):
        ecg_length = ecg_length_seconds * sr
        # pad zeros if needed
        if x.shape[1] < ecg_length:
            x = torch.cat(
                (x, torch.zeros((x.shape[0], ecg_length - x.shape[1]))), axis=1
            )

        if setting == "train":
            return T.RandomCrop(size=(x.shape[0], ecg_length))(x)
        else:
            return x[:, :ecg_length]

    @staticmethod
    def normalize_ecg(ecg):
        if isinstance(ecg, torch.Tensor):
            ecg = ecg.numpy()
        if len(ecg.shape) == 3 and ecg.shape[0] == 1:
            ecg = ecg[0]
        ecg = torch.tensor(process_ecg(ecg))
        return ecg

    # Add Gaussian noise
    @staticmethod
    def add_noise(x, p=0.3, noise_level=0.03):
        if random.random() > p:
            return x
        noise = torch.randn_like(x) * noise_level
        return x + noise

    # Random amplitude scaling
    @staticmethod
    def scale(x, p=0.3, scale_range=(0.8, 1.2)):
        if random.random() > p:
            return x
        s = random.uniform(*scale_range)
        return x * s

    # Time shift (roll)
    @staticmethod
    def time_shift(x, p=0.4, shift_max_sec=1, sr=500):
        if random.random() > p:
            return x
        shift = int(random.uniform(-shift_max_sec, shift_max_sec) * sr)
        return torch.roll(x, shifts=shift, dims=-1)

    # Random dropout of small segments
    @staticmethod
    def drop_segments(x, p=0.5, segment_max_sec=0.3, drop_cnt=5, sr=500):
        if random.random() > p:
            return x
        mask = torch.ones_like(x)
        length = x.shape[-1]
        seg_len = int(segment_max_sec * sr)
        for i in range(drop_cnt):
            start = random.randint(0, max(0, length - seg_len))
            mask[..., start : start + seg_len] = 0
        return x * mask

    # Low-pass filter via Fourier domain
    def lowpass(self, x, p=0.1):
        if random.random() > p:
            return x
        kernel_size = int(0.1 * self.sr)  # 100ms window
        kernel = torch.ones(1, 1, kernel_size, device=x.device) / kernel_size
        return torch.conv1d(x.unsqueeze(0), kernel, padding=kernel_size // 2).squeeze(0)

    def augment(self, x, sampling_rate=500):
        x = ECGTransforms.time_shift(
            x,
            p=self.config.time_shift.p,
            shift_max_sec=self.config.time_shift.shift_max_sec,
            sr=sampling_rate,
        )
        x = ECGTransforms.scale(
            x,
            p=self.config.amp_scale.p,
            scale_range=self.config.amp_scale.scale_range,
        )
        x = ECGTransforms.add_noise(
            x, p=self.config.noise.p, noise_level=self.config.noise.noise_level
        )
        x = ECGTransforms.drop_segments(
            x,
            p=self.config.drop.p,
            segment_max_sec=self.config.drop.segment_max_sec,
            drop_cnt=self.config.drop.drop_cnt,
            sr=sampling_rate,
        )
        return x

    # Compose random augmentations
    def __call__(self, x, sampling_rate, ecg_length_seconds):

        x = ECGTransforms.sample_ecg(x, sampling_rate, ecg_length_seconds, self.setting)
        assert tuple(x.shape) == (12, sampling_rate * ecg_length_seconds), tuple(
            x.shape
        )
        x = ECGTransforms.normalize_ecg(x).squeeze()
        if self.do_augmentation:
            x = self.augment(x, sampling_rate)
        return x
