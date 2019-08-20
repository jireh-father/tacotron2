import numpy as np
from scipy.io.wavfile import read
import torch
import librosa


def get_mask_from_lengths(lengths, add=0):
    max_len = torch.max(lengths).item() + add
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).byte()
    return mask


def load_wav_to_torch(full_path, sampling_rate):
    data = librosa.core.load(full_path, sr=sampling_rate)[0]
    data = data / np.abs(data).max() * 0.999
    return torch.FloatTensor(data.astype(np.float32))
    # sampling_rate, data = read(full_path)
    # return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)
