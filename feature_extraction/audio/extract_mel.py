import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import tqdm
import pyloudnorm as pyln
from skimage.transform import resize
import struct
import webrtcvad
from scipy.ndimage.morphology import binary_dilation
import librosa
import warnings


# 定义函数用于将wav文件转换为mel谱的矩阵
def wav_to_mel(wav_root, save_root):
    if not os.path.exists(save_root): os.makedirs(save_root)
    for wav_file in tqdm.tqdm(os.listdir(wav_root)):
        y, sr = librosa.load(os.path.join(wav_root, wav_file))
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)


def trim_long_silences(path, sr=None, return_raw_wav=False, norm=True, vad_max_silence_length=12):
    """
    Ensures that segments without voice in the waveform remain no longer than a
    threshold determined by the VAD parameters in params.py.
    :param wav: the raw waveform as a numpy array of floats
    :param vad_max_silence_length: Maximum number of consecutive silent frames a segment can have.
    :return: the same waveform with silences trimmed away (length <= original wav length)
    """

    ## Voice Activation Detection
    # Window size of the VAD. Must be either 10, 20 or 30 milliseconds.
    # This sets the granularity of the VAD. Should not need to be changed.
    int16_max = (2 ** 15) - 1

    sampling_rate = 16000
    wav_raw, sr = librosa.core.load(path, sr=sr)

    if norm:
        meter = pyln.Meter(sr)  # create BS.1770 meter
        loudness = meter.integrated_loudness(wav_raw)
        wav_raw = pyln.normalize.loudness(wav_raw, loudness, -20.0)
        if np.abs(wav_raw).max() > 1.0:
            wav_raw = wav_raw / np.abs(wav_raw).max()

    wav = librosa.resample(wav_raw, sr, sampling_rate, res_type='kaiser_best')

    vad_window_length = 30  # In milliseconds
    # Number of frames to average together when performing the moving average smoothing.
    # The larger this value, the larger the VAD variations must be to not get smoothed out.
    vad_moving_average_width = 8

    # Compute the voice detection window size
    samples_per_window = (vad_window_length * sampling_rate) // 1000

    # Trim the end of the audio to have a multiple of the window size
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]

    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))

    # Perform voice activation detection
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                         sample_rate=sampling_rate))
    voice_flags = np.array(voice_flags)

    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width

    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(np.bool)

    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)
    audio_mask = resize(audio_mask, (len(wav_raw),)) > 0
    if return_raw_wav:
        return wav_raw, audio_mask, sr
    return wav_raw[audio_mask], audio_mask, sr


def librosa_pad_lr(x, fsize, fshift, pad_sides=1):
    '''compute right padding (final frame) or both sides padding (first and final frames)
    '''
    assert pad_sides in (1, 2)
    # return int(fsize // 2)
    pad = (x.shape[0] // fshift + 1) * fshift - x.shape[0]
    if pad_sides == 1:
        return 0, pad
    else:
        return pad // 2, pad // 2 + pad % 2


def librosa_wav2spec(wav_path,
                     fft_size=1024,
                     hop_size=256,
                     win_length=1024,
                     window="hann",
                     num_mels=80,
                     fmin=80,
                     fmax=-1,
                     eps=1e-6,
                     sample_rate=22050,
                     loud_norm=False,
                     trim_long_sil=False):
    if isinstance(wav_path, str):
        if trim_long_sil:
            wav, _, _ = trim_long_silences(wav_path, sample_rate)
        else:
            wav, _ = librosa.core.load(wav_path, sr=sample_rate)
    else:
        wav = wav_path

    if loud_norm:
        meter = pyln.Meter(sample_rate)  # create BS.1770 meter
        loudness = meter.integrated_loudness(wav)
        wav = pyln.normalize.loudness(wav, loudness, -22.0)
        if np.abs(wav).max() > 1:
            wav = wav / np.abs(wav).max()

    # get amplitude spectrogram
    x_stft = librosa.stft(wav, n_fft=fft_size, hop_length=hop_size,
                          win_length=win_length, window=window, pad_mode="constant")
    linear_spc = np.abs(x_stft)  # (n_bins, T)

    # get mel basis
    fmin = 0 if fmin == -1 else fmin
    fmax = sample_rate / 2 if fmax == -1 else fmax
    mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=fft_size, n_mels=num_mels, fmin=fmin, fmax=fmax)

    # calculate mel spec
    mel = mel_basis @ linear_spc
    mel = np.log10(np.maximum(eps, mel))  # (n_mel_bins, T)
    l_pad, r_pad = librosa_pad_lr(wav, fft_size, hop_size, 1)
    wav = np.pad(wav, (l_pad, r_pad), mode='constant', constant_values=0.0)
    wav = wav[:mel.shape[1] * hop_size]

    # log linear spec
    linear_spc = np.log10(np.maximum(eps, linear_spc))
    return {'wav': wav, 'mel': mel.T, 'linear': linear_spc.T, 'mel_basis': mel_basis}


def process_audio(wav_root, save_root):
    for wav_file in tqdm.tqdm(os.listdir(wav_root)):
        wav_fn = os.path.join(wav_root, wav_file)
        wav2spec_dict = librosa_wav2spec(
            wav_fn,
            fft_size=1024,  # text2mel_params['fft_size'],       # 1024
            hop_size=256,  # text2mel_params['hop_size'],       # 256
            win_length=1024,  # text2mel_params['win_size'],     # 1024
            num_mels=256,  # text2mel_params['audio_num_mel_bins'],     # 80
            fmin=55,  # text2mel_params['fmin'],       # 55
            fmax=7600,  # text2mel_params['fmax'],       # 7600
            sample_rate=22050,  # text2mel_params['audio_sample_rate'],       # 22050
            loud_norm=False)  # text2mel_params['loud_norm'])     # False
        mel = wav2spec_dict['mel']
        # wav = wav2spec_dict['wav'].astype(np.float16)
        # res.update({'mel': mel, 'wav': wav, 'sec': len(wav) / text2mel_params['audio_sample_rate'], 'len': mel.shape[0]})
        # res.update({'mel': mel, 'wav': wav, 'sec': len(wav) / 22050, 'len': mel.shape[0]})
        # return wav, mel

        # 保存mel特征为.npy文件，名字与wav文件相同
        npy_file = os.path.splitext(wav_file)[0] + '.npy'
        np.save(os.path.join(save_root, npy_file), mel)


if __name__ == '__main__':
    import fire

    fire.Fire()

# # 调用函数实现wav文件到mel谱的转换
# wav_file = "your_wav_file.wav"
# wav_to_mel(wav_file)
