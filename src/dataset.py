import os
import numpy as np
import librosa
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import pickle

class MusicDataset:
    """Dataset class for processing music audio files"""
    
    def __init__(self, data_dir='data/audio/genres', sr=22050, n_mfcc=20, 
                 max_duration=30, feature_type='mfcc'):
        self.data_dir = Path(data_dir)
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.max_duration = max_duration
        self.feature_type = feature_type
        self.features = []
        self.labels = []
        self.filenames = []
        self.genre_mapping = {}
        
    def extract_features(self, audio_path):
        try:
            y, sr = librosa.load(audio_path, sr=self.sr, duration=self.max_duration)
            
            if self.feature_type == 'mfcc':
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
                features = np.mean(mfcc, axis=1)
            elif self.feature_type == 'melspectrogram':
                mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                features = np.mean(mel_spec_db, axis=1)
            else:
                raise ValueError(f"Unknown feature type: {self.feature_type}")
            
            return features
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None
    
    def load_dataset(self):
        print(f"Loading dataset from {self.data_dir}")
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        genre_folders = [f for f in self.data_dir.iterdir() if f.is_dir()]
        
        if len(genre_folders) == 0:
            raise ValueError(f"No genre folders found in {self.data_dir}")
        
        self.genre_mapping = {genre.name: idx for idx, genre in enumerate(sorted(genre_folders))}
        print(f"Found {len(self.genre_mapping)} genres: {list(self.genre_mapping.keys())}")
        
        for genre_folder in tqdm(sorted(genre_folders), desc="Processing genres"):
            genre_name = genre_folder.name
            genre_label = self.genre_mapping[genre_name]
            
            audio_files = list(genre_folder.glob('*.wav')) + list(genre_folder.glob('*.mp3'))
            
            for audio_file in tqdm(audio_files, desc=f"  {genre_name}", leave=False):
                features = self.extract_features(audio_file)
                
                if features is not None:
                    self.features.append(features)
                    self.labels.append(genre_label)
                    self.filenames.append(str(audio_file))
        
        self.features = np.array(self.features)
        self.labels = np.array(self.labels)
        
        print(f"\nDataset loaded successfully!")
        print(f"Total samples: {len(self.features)}")
        print(f"Feature shape: {self.features.shape}")
        print(f"Labels shape: {self.labels.shape}")
        
        return self.features, self.labels
    
    def save_processed_data(self, save_path='data/processed_features.pkl'):
        data = {
            'features': self.features,
            'labels': self.labels,
            'filenames': self.filenames,
            'genre_mapping': self.genre_mapping,
            'params': {
                'sr': self.sr,
                'n_mfcc': self.n_mfcc,
                'max_duration': self.max_duration,
                'feature_type': self.feature_type
            }
        }
        
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Processed data saved to {save_path}")
    
    @staticmethod
    def load_processed_data(load_path='data/processed_features.pkl'):
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Loaded processed data from {load_path}")
        print(f"Features shape: {data['features'].shape}")
        print(f"Labels shape: {data['labels'].shape}")
        print(f"Genres: {list(data['genre_mapping'].keys())}")
        
        return data


def normalize_features(features):
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    std[std == 0] = 1
    normalized = (features - mean) / std
    return normalized, mean, std


if __name__ == "__main__":
    print("=" * 60)
    print("GTZAN Dataset Processing")
    print("=" * 60)
    
    dataset = MusicDataset(
        data_dir='data/audio/genres',
        sr=22050,
        n_mfcc=20,
        max_duration=30,
        feature_type='mfcc'
    )
    
    features, labels = dataset.load_dataset()
    features_normalized, mean, std = normalize_features(features)
    dataset.features = features_normalized
    dataset.save_processed_data('data/processed_features.pkl')
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)