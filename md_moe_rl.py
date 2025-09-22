import sys
import os
from datetime import datetime
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy import signal

def load_audio(audio_path, sr=16000, chunk_duration=3):
    """
    音频特征提取函数 | Audio feature extraction
    
    计算三种声学特征：
    1. Mel频谱（使用Librosa）
    2. Welch功率谱（使用SciPy）
    3. 平均幅度谱（FFT）
    
    Computes three acoustic features:
    1. Mel-spectrogram (using Librosa)
    2. Welch power spectrum (using SciPy)
    3. Average amplitude spectrum (FFT)
    
    Args:
        audio_path: 音频文件路径 | Audio file path
        sr: 采样率 (默认16000) | Sampling rate (default 16000)
        chunk_duration: 截取时长(秒) | Truncation duration in seconds
    
    Returns:
        tuple: (梅尔频谱, 韦尔奇谱, 平均幅度谱)
              (Mel-spectrogram, Welch spectrum, Average amplitude spectrum)
    """
    wav, sr_ret = librosa.load(audio_path.strip(), sr=sr)
    num_wav_samples = wav.shape[0]
    num_chunk_samples = int(chunk_duration * sr)
    if num_wav_samples > num_chunk_samples:
        wav = wav[:num_chunk_samples]
   
    # features = librosa.feature.mfcc(yy=wav, sr=sr,n_mfcc=80, hop_length=512)    #(80, 157)
    # log mel
    features = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=1024, n_mels=200, hop_length=160, win_length=400)#(80, 501)
    features = librosa.power_to_db(features, ref=1.0, amin=1e-10, top_db=None)#(80, 501)

    # 计算韦尔奇谱
    freqs, welch_spectrum = signal.welch(wav, fs=sr, nperseg=400)
    
    welch_spectrum = 10 * np.log10(welch_spectrum)  # 转换为dB

    # 计算平均幅度谱
    avg_amp_spectrum = np.abs(np.fft.rfft(wav, n=400))
    avg_amp_spectrum = 10 * np.log10(avg_amp_spectrum+1)  # 转换为dB

    return features, welch_spectrum, avg_amp_spectrum  # 返回梅尔频谱和韦尔奇谱

def load_saved_features(features_path):
    if os.path.exists(features_path):
        return np.load(features_path)
    return None

class Dataset_audio(Dataset):
    """
    音频数据集类 | Audio dataset class
    
    特征缓存机制：
    1. 优先加载预计算的特征文件（.npy）
    2. 特征缺失时实时计算并保存
    3. 异常时自动重试随机样本
    
    Feature caching mechanism:
    1. Load pre-computed features first
    2. Compute and save when missing
    3. Auto-retry with random sample on error
    """
    def __init__(self, data_list_path, sr=16000, chunk_duration=5):
        super(Dataset_audio, self).__init__()
        with open(data_list_path, 'r') as f:
            self.lines = f.readlines()
        self.sr = sr
        self.chunk_duration = chunk_duration

    def __getitem__(self, idx):
        try:
            audio_path, label, Rr, Sz = self.lines[idx].replace('\n', '').split('\t')
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            features_path_mel = os.path.join(os.path.dirname(audio_path), 'features', f"{base_name}_mel.npy")
            features_path_wel = os.path.join(os.path.dirname(audio_path), 'features', f"{base_name}_wel.npy")
            features_path_avg = os.path.join(os.path.dirname(audio_path), 'features', f"{base_name}_avg.npy")

            spec_mag = load_saved_features(features_path_mel)
            welch_spec = load_saved_features(features_path_wel)
            avg_amp_spectrum = load_saved_features(features_path_avg)
            if spec_mag is None or welch_spec is None or avg_amp_spectrum is None:
                # 构建特征保存路径，与 wav 文件同目录
                base_name = os.path.splitext(os.path.basename(audio_path))[0]
                features_path_mel = os.path.join(os.path.dirname(audio_path), 'features', f"{base_name}_mel.npy")
                features_path_wel = os.path.join(os.path.dirname(audio_path), 'features', f"{base_name}_wel.npy")
                features_path_avg = os.path.join(os.path.dirname(audio_path), 'features', f"{base_name}_avg.npy")
                # 加载音频并计算特征
                spec_mag, welch_spec, avg_amp_spectrum = load_audio(audio_path, sr=self.sr, chunk_duration=self.chunk_duration)
                # 如果不存在目录，则创建
                if not os.path.exists(os.path.dirname(features_path_mel)):
                    os.makedirs(os.path.dirname(features_path_mel))
                # 保存特征到 wav 文件同目录
                np.save(features_path_mel, spec_mag)
                np.save(features_path_wel, welch_spec)
                np.save(features_path_avg, avg_amp_spectrum)

            welch_spec = welch_spec.astype('float32')
            spec_mag = spec_mag.astype('float32')
            avg_amp_spectrum = avg_amp_spectrum.astype('float32')
            return spec_mag, welch_spec, avg_amp_spectrum, np.array(int(label), dtype=np.int64), np.array(Rr, dtype=np.float64), np.array(Sz, dtype=np.float64)
        except Exception as ex:
            print(f"[{datetime.now()}] 数据: {self.lines[idx]} 出错，错误信息: {ex}", file=sys.stderr)
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)

    def __len__(self):
        return len(self.lines)

# 使用示例
if __name__ == "__main__":
    data_list_path = "/home/wangtengbo/A800-7-nfs/A800-13-data/temp_demo/UWTRL-MEG-main/data/test_list.txt"
    dataset = Dataset_audio(data_list_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for spec_mag, welch_spec, avg_amp_spectrum, label, Rr, Rz in dataloader:
        # 打印
        print(spec_mag.shape)
        print(welch_spec.shape)
        print(avg_amp_spectrum.shape)
        print(label.shape)
        print(Rr.shape)
        print(Rz.shape)