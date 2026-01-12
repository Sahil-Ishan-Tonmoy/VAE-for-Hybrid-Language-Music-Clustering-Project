# src/feature_loader.py
import pickle


def load_features(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    if 'latent_features' in data:
        features = data['latent_features']
    elif 'features' in data:
        features = data['features']
    else:
        raise KeyError(f"No feature key found in {filepath}")

    labels = data['labels']
    return features, labels


def load_with_metadata(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)
