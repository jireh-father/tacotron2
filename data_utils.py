import random
import numpy as np
import torch
import torch.utils.data

import layers
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence
import os
import pickle

class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """

    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.speaker_embedding_dir = hparams.speaker_embedding_dir
        self.mel_dir = hparams.mel_dir
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)
        self.use_model_speaker_embedding = hparams.use_model_speaker_embedding
        if not hparams.use_model_speaker_embedding:
            self.spk_id_map = pickle.load(open(self.speaker_embedding_dir, "rb"))
            self.nums_of_speakers = hparams.nums_of_speakers

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath = os.path.join(self.mel_dir, os.path.basename(audiopath_and_text[0])) + ".pt"
        text = audiopath_and_text[1]
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        if self.use_model_speaker_embedding:
            speaker_embedding_path = os.path.join(self.speaker_embedding_dir, os.path.basename(audiopath_and_text[0])) + ".npy"
            speaker_embedding = self.get_speaker_embedding(speaker_embedding_path)
        else:
            spk_file_name = os.path.basename(audiopath_and_text[0]).split(".")[0]
            speaker_embedding = self.spk_id_map[spk_file_name]
        return (text, mel, speaker_embedding)

    def get_speaker_embedding(self, filename):
        speaker_embedding_np = np.load(filename)
        speaker_embedding_np = torch.autograd.Variable(torch.FloatTensor(speaker_embedding_np.astype(np.float32)), requires_grad=False)
        # speaker_embedding_np = speaker_embedding_np.half() if self.is_fp16 else speaker_embedding_np
        return speaker_embedding_np

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            # audio_norm = load_wav_to_torch(filename, self.stft.sampling_rate)
            # audio, sampling_rate = load_wav_to_torch(filename)
            # if sampling_rate != self.stft.sampling_rate:
            #     raise ValueError("{} {} SR doesn't match target {} SR".format(
            #         sampling_rate, self.stft.sampling_rate))
            # audio_norm = audio / self.max_wav_value

            melspec = torch.load(filename)
            # mel = torch.autograd.Variable(mel.cuda())
            # mel = torch.unsqueeze(mel, 0)
            # melspec = mel.half() if self.is_fp16 else melsss

            # audio_norm = audio_norm.unsqueeze(0)
            # audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            # melspec = self.stft.mel_spectrogram(audio_norm)
            # melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """

    def __init__(self, n_frames_per_step, use_model_speaker_embedding):
        self.n_frames_per_step = n_frames_per_step
        self.use_model_speaker_embedding = use_model_speaker_embedding

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        if self.use_model_speaker_embedding:
            speaker_embeddings = torch.FloatTensor(len(batch), batch[0][2].size(0))
        else:
            speaker_embeddings = torch.IntTensor(len(batch), 0)
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1) - 1:] = 1
            output_lengths[i] = mel.size(1)
            speaker_embeddings[i] = batch[ids_sorted_decreasing[i]][2]

        return text_padded, input_lengths, mel_padded, gate_padded, \
               output_lengths, speaker_embeddings
