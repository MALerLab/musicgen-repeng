import torch
from tqdm import tqdm
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA

def project_onto_direction(H, direction):
    """Project matrix H (n, d_1) onto direction vector (d_2,)"""
    mag = np.linalg.norm(direction)
    assert not np.isinf(mag)
    return (H @ direction) / mag

def get_directions(path):
    representation_pairs = []
    for p in tqdm(Path(path).rglob('*.pt')):
        loaded = torch.load(str(p), map_location=torch.device('cpu'))[-1]
        representation_pairs.append(loaded)

    representation_pairs = torch.cat(representation_pairs, dim=0)
    representation_pairs = representation_pairs.permute(1,0,2)

    relative_layer_hiddens = {}
    for layer, pair in enumerate(representation_pairs):
        relative_layer_hiddens[layer] = (
            pair[::2] - pair[1::2]
        )

    directions = {}
    for layer in range(len(relative_layer_hiddens)):
        # assert representation_pairs[layer].shape[0] == 110 * 2

        # fit layer directions
        train = np.vstack(
            relative_layer_hiddens[layer].to("cpu").numpy()
            - relative_layer_hiddens[layer].to("cpu").numpy().mean(axis=0, keepdims=True)
        )
        pca_model = PCA(n_components=1, whiten=False).fit(train)
        # shape (n_features,)
        directions[layer] = pca_model.components_.astype(np.float32).squeeze(axis=0)

        # calculate sign
        projected_hiddens = project_onto_direction(
            representation_pairs[layer].to("cpu").numpy(), directions[layer]
        )

        # order is [positive, negative, positive, negative, ...]
        positive_smaller_mean = np.mean(
            [
                projected_hiddens[i] < projected_hiddens[i + 1]
                for i in range(0, representation_pairs.shape[1], 2)
            ]
        )
        positive_larger_mean = np.mean(
            [
                projected_hiddens[i] > projected_hiddens[i + 1]
                for i in range(0, representation_pairs.shape[1], 2)
            ]
        )

        if positive_smaller_mean > positive_larger_mean:  # type: ignore
            directions[layer] *= -1

    return directions

def get_directions_each_layer(path):
    directions = {}

    for layer in tqdm(range(48)):
        representation_pairs = []
        for p in tqdm(Path(path).rglob('*.pt')):
            loaded = torch.load(str(p), map_location=torch.device('cpu'))[:,layer,:]
            representation_pairs.append(loaded)

        representation_pairs = torch.cat(representation_pairs, dim=0)

        relative_layer_hiddens = (
            representation_pairs[::2] - representation_pairs[1::2]
        )

        # assert representation_pairs[layer].shape[0] == 110 * 2

        # fit layer directions
        train = np.vstack(
            relative_layer_hiddens.to("cpu").numpy()
            - relative_layer_hiddens.to("cpu").numpy().mean(axis=0, keepdims=True)
        )
        pca_model = PCA(n_components=1, whiten=False).fit(train)
        # shape (n_features,)
        directions[layer] = pca_model.components_.astype(np.float32).squeeze(axis=0)

        # calculate sign
        projected_hiddens = project_onto_direction(
            representation_pairs.to("cpu").numpy(), directions[layer]
        )

        # order is [positive, negative, positive, negative, ...]
        positive_smaller_mean = np.mean(
            [
                projected_hiddens[i] < projected_hiddens[i + 1]
                for i in range(0, representation_pairs.shape[1], 2)
            ]
        )
        positive_larger_mean = np.mean(
            [
                projected_hiddens[i] > projected_hiddens[i + 1]
                for i in range(0, representation_pairs.shape[1], 2)
            ]
        )

        if positive_smaller_mean > positive_larger_mean:  # type: ignore
            directions[layer] *= -1

    return directions

directions = get_directions_each_layer('/home/sake/MusicGenRepEng_Dataset_conti60ms_energy_mediummodel_norm_nob4layer')

torch.save(directions, '/home/sake/MusicGenRepEng_Dataset_conti60ms_energy_mediummodel_norm_nob4layer_directions.pth')