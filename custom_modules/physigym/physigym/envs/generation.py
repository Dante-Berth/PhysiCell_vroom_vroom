import numpy as np
import pandas as pd
from tysserand import tysserand as ty
from collections import defaultdict
import os
import shutil

from math import sqrt
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib
import random

matplotlib.use("Agg")


def set_global_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)


def generate_correlated_field(
    domain_size,
    correlation_length,
    key,
):
    """
    Generate a 2D Gaussian-correlated random field.

    Parameters
    ----------
    shape : tuple
        Field dimensions (height, width).
    correlation_length : float
        Spatial correlation length (Gaussian kernel std).

    Returns
    -------
    field : ndarray
        2D correlated scalar field.
    """

    noise = np.random.randn(*domain_size)
    sigma = correlation_length / sqrt(2)
    field = gaussian_filter(noise, sigma=sigma, mode="reflect")

    return field


def generate_balanced_fields(
    domain_size,
    params,
    amplitude=1.0,
    local_noise_level=0.3,
):
    """
    Génère un champ global spatialement corrélé + un bruit local filtré pour chaque type.

    Parameters:
    -----------
    shape : tuple
        Dimensions du champ (height, width)
    cell_types : list
        Liste des types cellulaires
    correlation_length : float
        Longueur de corrélation spatiale
    amplitude : float
        Amplitude globale du champ
    local_noise_level : float
        Intensité du bruit local relatif

    Returns:
    --------
    fields : dict
        Dictionnaire {type_cellulaire: champ 2D}
    """

    local_noise_level = local_noise_level / 100 + 0.05

    fields = {}

    for key in params.keys():
        correlation_length = params[key]["correlation_length"]
        # === Génère un bruit local filtré ===
        local_noise = np.random.randn(*domain_size)
        filtered_noise = gaussian_filter(local_noise, sigma=correlation_length / 3)

        # === Champ final : base + bruit doux ===
        final_field = amplitude * (
            generate_correlated_field(domain_size, correlation_length, key=key)
            + local_noise_level * filtered_noise
        )
        min_final_field = np.min(final_field)
        max_final_field = np.max(final_field)
        fields[key] = (max_final_field - final_field) / (
            max_final_field - min_final_field
        )

    return fields


def weighted_pick(arr, threshold, n=1):
    # mask only the values above threshold
    mask = arr > threshold

    # get coordinates of valid pixels
    coords = np.argwhere(mask)

    # get the corresponding probabilities
    probs = arr[mask].astype(float)

    # normalize probabilities to sum to 1
    probs /= probs.sum()

    # weighted choice
    idx = np.random.choice(len(coords), size=n, p=probs, replace=False)

    return coords[idx]


def generate_synthetic_network_field(
    params,
    x_min,
    x_max,
    y_min,
    y_max,
    amplitude=1,
    save=False,
):
    domain_size = int(x_max - x_min), int(y_max - y_min)
    # === Generate Fields ===
    fields = generate_balanced_fields(
        domain_size=domain_size,
        params=params,
        amplitude=amplitude,
    )
    xs_final = []
    ys_final = []
    phenotypes_final = []
    n_types = len(list(params.keys()))
    fig, axes = (
        plt.subplots(n_types, 2, figsize=(10, 5 * n_types)) if save else None,
        None,
    )

    # Handle case of single row
    if n_types == 1:
        axes = np.array([axes])

    for row_idx, ct in enumerate(list(params.keys())):
        field = fields[ct]
        coords = weighted_pick(
            field, threshold=params[ct]["threshold"], n=params[ct]["number_cells"]
        )
        xs = coords[:, 0]
        ys = coords[:, 1]
        if save:
            # ========== LEFT: FIELD ==========
            ax_field = axes[row_idx, 0]
            im = ax_field.imshow(field, cmap="viridis")
            ax_field.set_title(f"Field: {ct}")
            ax_field.axis("off")
            fig.colorbar(im, ax=ax_field, fraction=0.046, pad=0.04)

            # ========== RIGHT: SCATTER CELLS ==========
            ax_scatter = axes[row_idx, 1]
            ax_scatter.scatter(ys, domain_size[1] - xs, s=10, alpha=0.8)
            ax_scatter.set_title(f"Cells: {ct}")
            ax_scatter.set_xlabel("X")
            ax_scatter.set_ylabel("Y")
            ax_scatter.set_aspect("equal")

        xs_final.extend(xs)  # extend, not append
        ys_final.extend(ys)
        phenotypes_final.extend([ct] * len(coords))  # repeat ct for each cell

    df_cells = pd.DataFrame(
        data={
            "x": xs_final,
            "y": ys_final,
            "z": [0] * len(xs_final),
            "type": phenotypes_final,
        }
    )
    df_cells[["x", "y"]] += x_min, y_min

    return df_cells


def network_field(
    params,
    x_min,
    x_max,
    y_min,
    y_max,
    name_folder,
    i,
    amplitude=1,
):
    df_cells = generate_synthetic_network_field(
        params=params,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        amplitude=amplitude,
        save=False,
    )

    df_cells.to_csv(f"./{name_folder}/df_{i}.csv", index=False)
    df_cells["typeID"] = df_cells["type"].astype("category").cat.codes

    plt.figure(figsize=(8, 8))

    scatter = plt.scatter(
        df_cells["x"],
        df_cells["y"],
        c=df_cells["typeID"],
        cmap="tab10",  # Pick a nice categorical colormap
        alpha=0.8,
        s=20,
    )

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Cell Positions by Phenotype")

    # Add legend mapped to phenotype categories
    handles, labels = scatter.legend_elements(prop="colors", alpha=0.8)
    plt.legend(handles, labels, title="Phenotypes")

    plt.gca().set_aspect("equal")
    plt.savefig(f"./{name_folder}/df_{i}.png")


def create_csv(seed, cell_2_fraction, csv_path, params, x_min, x_max, y_min, y_max):
    if seed is not None:
        set_global_seed(seed)
    else:
        set_global_seed(42)

    df = generate_synthetic_network_field(
        params=params,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        amplitude=1,
        save=False,
    )

    if cell_2_fraction is None:
        cell_2_fraction = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    cell_2_fraction = (
        np.random.choice(cell_2_fraction)
        if isinstance(cell_2_fraction, (list, np.ndarray))
        else cell_2_fraction
    )
    cell1_indices = df[df["type"] == "cell_1"].index
    n_to_change = int(cell_2_fraction * len(cell1_indices))
    if n_to_change > 0:
        indices_to_change = np.random.choice(cell1_indices, n_to_change)
        df.loc[indices_to_change, "type"] = "cell_2"
    df["z"] = 0.0
    # Drop trailing empty columns
    while df.iloc[:, -1].isna().all() or (df.iloc[:, -1] == "").all():
        df = df.iloc[:, :-1]

    df = df.drop_duplicates(subset=["x", "y"], keep=False)

    df.to_csv(csv_path, index=False, float_format="%.6f")


if __name__ == "__main__":
    # === Global parameter ===

    # === Network parameter ===
    x_min, x_max, y_min, y_max = -256, 256, -256, 256
    name_folder = "config_network_field"
    params = {
        "tumor": {"correlation_length": 45, "threshold": 0.55, "number_cells": 512},
        "cell_1": {"correlation_length": 45, "threshold": 0.55, "number_cells": 128},
    }

    d_arg_generation = {
        "cell_2_fraction": 0.3,
        "params": params,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "seed": 42,
        "csv_path": f"./{name_folder}/df.csv",
    }

    os.makedirs(f"./{name_folder}", exist_ok=True)
    create_csv(**d_arg_generation)
    from multiprocessing import Pool, cpu_count
    from tqdm import tqdm

    seed = 42

    def wrapper(i):
        return network_field(
            params=params,
            name_folder=name_folder,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            i=i,
            amplitude=1,
            seed=seed + i,
        )

    N = 10
    num_workers = cpu_count()  # or set manually

    with Pool(num_workers) as p:
        list(tqdm(p.imap_unordered(wrapper, range(N)), total=N))
    for i in tqdm(range(10)):
        wrapper(i)
