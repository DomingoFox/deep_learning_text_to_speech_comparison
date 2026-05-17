import numpy as np
import soundfile as sf
import json


def compute_duration(audio, sr):
    if sr <= 0:
        raise ValueError('Sample rate must be positive.')
    return len(audio) / sr


def compute_spectral_features(audio, sr=16000):
    import librosa

    audio = prepare_audio(audio)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

    return {
        'spectral_centroid': float(np.mean(spectral_centroid)),
        'spectral_rolloff': float(np.mean(spectral_rolloff)),
        'mfcc_std': float(np.std(mfcc))
    }


def compute_loudness(audio):
    audio = prepare_audio(audio)
    rms_energy = np.sqrt(np.mean(audio ** 2))
    return float(rms_energy)


def compute_zero_crossing_rate(audio):
    import librosa

    audio = prepare_audio(audio)
    zcr = librosa.feature.zero_crossing_rate(audio)
    return float(np.mean(zcr))


def compute_temporal_variance(audio, sr=16000, hop_length=512):
    import librosa

    audio = prepare_audio(audio)
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop_length)
    if len(onset_env) < 2:
        return 0.0
    flux = np.sqrt(np.sum(np.diff(onset_env)**2))
    return float(flux / len(onset_env))


def prepare_audio(audio):
    audio = np.asarray(audio).squeeze()
    if audio.size == 0:
        raise ValueError('Audio array is empty.')
    return audio.astype(np.float32)


def normalize_audio(audio):
    audio = prepare_audio(audio)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio / max_val
    return audio


def save_audio(audio, sr, filepath):
    audio = prepare_audio(audio)
    sf.write(filepath, audio, sr)


def load_audio(filepath, sr=16000):
    import librosa

    audio, _ = librosa.load(filepath, sr=sr)
    return audio


class MetricsCollector:
    def __init__(self):
        self.results = {}

    def add_model_result(self, model_name, prompt, inference_time, audio, sr=16000):
        if model_name not in self.results:
            self.results[model_name] = []

        spectral = compute_spectral_features(audio, sr)

        result = {
            'prompt': prompt,
            'inference_time': inference_time,
            'duration': compute_duration(audio, sr),
            'loudness': compute_loudness(audio),
            'spectral_centroid': spectral['spectral_centroid'],
            'spectral_rolloff': spectral['spectral_rolloff'],
            'mfcc_std': spectral['mfcc_std'],
            'zero_crossing_rate': compute_zero_crossing_rate(audio),
            'temporal_variance': compute_temporal_variance(audio, sr)
        }

        self.results[model_name].append(result)

    def get_summary(self, model_name):
        if model_name not in self.results or len(self.results[model_name]) == 0:
            return {}

        data = self.results[model_name]

        summary = {
            'avg_inference_time': np.mean([d['inference_time'] for d in data]),
            'avg_duration': np.mean([d['duration'] for d in data]),
            'avg_loudness': np.mean([d['loudness'] for d in data]),
            'avg_spectral_centroid': np.mean([d['spectral_centroid'] for d in data]),
            'avg_spectral_rolloff': np.mean([d['spectral_rolloff'] for d in data]),
            'avg_mfcc_std': np.mean([d['mfcc_std'] for d in data]),
            'avg_zero_crossing_rate': np.mean([d['zero_crossing_rate'] for d in data]),
            'avg_temporal_variance': np.mean([d['temporal_variance'] for d in data])
        }

        return summary

    def save_results(self, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)

    def load_results(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            self.results = json.load(f)
