import os
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score, adjusted_rand_score

from generate_seq_embedding import generate_seq_embedding


def scatter_labeled_z(
    z_batch,
    colors,
    filename="test_plot.png",
    legend_labels=None,
    legend_title=None,
):
    """
    Save a 2D scatter plot of embeddings colored by class.

    Parameters
    ----------
    z_batch : np.ndarray, shape [N, 2]
        2D embeddings (e.g., t-SNE output).
    colors : list[str]
        Per-point matplotlib color strings.
    filename : str
        Output path for the PNG file.
    legend_labels : dict[str, str], optional
        Mapping {label_str: color_str}.
    legend_title : str, optional
        Title shown on legend.
    """
    plt.switch_backend("Agg")
    fig = plt.gcf()
    fig.set_size_inches(4, 4)
    plt.clf()

    for n in range(z_batch.shape[0]):
        plt.scatter(
            z_batch[n, 0],
            z_batch[n, 1],
            c=colors[n],
            s=50,
            marker="o",
            edgecolors="none",
        )

    ax = plt.subplot(111)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.xlabel("z1")
    plt.ylabel("z2")

    if legend_labels:
        handles = []
        for label_str, color_str in legend_labels.items():
            handles.append(
                Line2D(
                    [0], [0],
                    marker="o",
                    linestyle="",
                    markersize=6,
                    markerfacecolor=color_str,
                    markeredgecolor=color_str,
                    label=label_str,
                )
            )
        ax.legend(
            handles=handles,
            title=legend_title,
            loc="best",
            fontsize=6,
            title_fontsize=7,
            frameon=True,
        )

    plt.savefig(filename, bbox_inches="tight", dpi=800)


def read_cath_fasta_like_original(cath_fasta_path: str):
    """
    EXACTLY mimic the original evaluate_with_cath_more() reading style:

    Original code:
      input = open(cathpath)
      for line in input:
          labels.append(line[1:].split("|")[0].strip())
          line = next(input)
          seqs.append(line.strip())

    Assumption: sequences are single-line (same as original).
    """
    labels = []
    seqs = []

    with open(cath_fasta_path, "r") as f:
        for line in f:
            if not line.startswith(">"):
                continue
            labels.append(line[1:].split("|")[0].strip())  # e.g. "1.10.10.2080"
            seq_line = next(f)  # original assumes sequence is on the next line
            seqs.append(seq_line.strip())

    return labels, seqs


def evaluate_with_cath_more_seq_aligned(
    out_figure_path: str,
    steps: int,
    cath_fasta_path: str,
    config_path: str,
    checkpoint_path: str,
    truncate_inference=None,
    max_length_inference=None,
    afterproject: bool = False,
):
    """
    Sequence-side CATH clustering evaluation aligned to the ORIGINAL evaluate_with_cath_more() protocol,
    with an added legend on the plot.

    Returns
    -------
    scores : list[float]
        [CH_full_1, CH_tsne_1, ARI_1, sil_1,
         CH_full_2, CH_tsne_2, ARI_2, sil_2,
         CH_full_3, CH_tsne_3, ARI_3, sil_3]
    """
    CATH_CLASS_NAME = {
        "1": "Mainly Alpha",
        "2": "Mainly Beta",
        "3": "Alpha Beta",
    }

    CATH_ARCHI_NAME = {
        "3.30": "2-Layer Sandwich",
        "3.40": "3-Layer(aba) Sandwich",
        "1.10": "Orthogonal Bundle",
        "3.10": "Roll",
        "2.60": "Sandwich",
    }

    CATH_FOLD_NAME = {
        "1.10.10": "Arc Repressor Mutant, subunit A",
        "3.30.70": "Alpha-Beta Plaits",
        "2.60.40": "Immunoglobulin-like",
        "2.60.120": "Jelly Rolls",
        "3.40.50": "Rossmann fold",
    }

    Path(out_figure_path).mkdir(parents=True, exist_ok=True)

    # --- Read labels & sequences using the original logic ---
    labels, seqs = read_cath_fasta_like_original(cath_fasta_path)

    # --- Generate embeddings using generate_seq_embedding (your required API) ---
    # Keep insertion order so embeddings correspond to seqs order
    query_sequence = {str(i): str(seqs[i]) for i in range(len(seqs))}
    query_embedding_dic = generate_seq_embedding(
        query_sequence=query_sequence,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        truncate_inference=truncate_inference,
        max_length_inference=max_length_inference,
        afterproject=afterproject,
        residue_level=False,
    )

    # --- Build embeddings array (N, D) in insertion order ---
    vals = list(query_embedding_dic.values())
    seq_embeddings = np.vstack([v.reshape(1, -1) for v in vals])
    seq_embeddings = np.asarray(seq_embeddings)

    # --- Original debug distance stats (first 50) ---
    distances = torch.cdist(
        torch.tensor(seq_embeddings[:50, :], dtype=torch.float64),
        torch.tensor(seq_embeddings[:50, :], dtype=torch.float64),
    )
    print(f"cath Min distance: {distances.min()}")
    print(f"cath Max distance: {distances.max()}")
    print(f"cath Mean distance: {distances.mean()}")
    print(f"cath Std distance: {distances.std()}")

    # --- Original t-SNE settings ---
    mdel = TSNE(n_components=2, random_state=0, init="random", method="exact")
    z_tsne_seq = mdel.fit_transform(seq_embeddings)

    scores = []

    # --- Original hard-coded subsets ---
    tol_archi_seq = {"3.30", "3.40", "1.10", "3.10", "2.60"}
    tol_fold_seq = {"1.10.10", "3.30.70", "2.60.40", "2.60.120", "3.40.50"}

    for digit_num in [1, 2, 3]:
        color = []
        colorid = []
        keys = {}
        colorindex = 0

        # Original palettes
        if digit_num == 1:
            ct = [
                "blue", "red", "black", "yellow", "orange", "green", "olive", "gray",
                "magenta", "hotpink", "pink", "cyan", "peru", "darkgray", "slategray", "gold",
            ]
        else:
            ct = [
                "black", "yellow", "orange", "green", "olive", "gray",
                "magenta", "hotpink", "pink", "cyan", "peru", "darkgray", "slategray", "gold",
            ]

        select_index = []
        color_dict = {}

        # NOTE: This loop replicates the original behavior as closely as possible.
        for label in labels:
            key = ".".join([x for x in label.split(".")[0:digit_num]])

            if digit_num == 2:
                if key not in tol_archi_seq:
                    continue
            if digit_num == 3:
                if key not in tol_fold_seq:
                    continue

            if key not in keys:
                keys[key] = colorindex
                colorindex += 1

            # Original uses ct[(id) % len(ct)] for point color
            color.append(ct[(keys[key]) % len(ct)])
            colorid.append(keys[key])

            # Original uses labels.index(label) (may be buggy if duplicates exist, but we keep it)
            select_index.append(labels.index(label))

            # Original sets color_dict using ct[keys[key]] (no modulo); safe here since small #classes
            # Keep this aligned.
            color_dict[keys[key]] = ct[keys[key]]

        print(f"sample num (digit {digit_num}) = {len(select_index)}")
        if len(select_index) == 0:
            scores.extend([float("nan")] * 4)
            continue

        # --- Legend mapping (added): show selected keys with their colors ---
        legend_title = {1: "CATH class", 2: "CATH architecture", 3: "CATH fold"}[digit_num]

        if digit_num == 1:
            code2name = CATH_CLASS_NAME
        elif digit_num == 2:
            code2name = CATH_ARCHI_NAME
        else:
            code2name = CATH_FOLD_NAME

        legend_labels = {
            f"{code} ({code2name.get(code, 'Unknown')})": ct[(idx) % len(ct)]
            for code, idx in keys.items()
        }

        # --- Scores aligned to original (uses 'color' list as labels) ---
        scores.append(calinski_harabasz_score(seq_embeddings[select_index], color))
        scores.append(calinski_harabasz_score(z_tsne_seq[select_index], color))

        # --- Plot (aligned filename) ---
        scatter_labeled_z(
            z_tsne_seq[select_index],
            color,
            filename=os.path.join(out_figure_path, f"step_{steps}_CATH_{digit_num}.png"),
            legend_labels=legend_labels,
            legend_title=legend_title,
        )

        # --- KMeans on t-SNE + ARI (aligned) ---
        kmeans = KMeans(n_clusters=len(color_dict), random_state=42)
        predicted_labels = kmeans.fit_predict(z_tsne_seq[select_index])
        ari = adjusted_rand_score(colorid, predicted_labels)
        scores.append(ari)

        # --- Silhouette on full embeddings (aligned: uses 'color' list as labels) ---
        scores.append(silhouette_score(seq_embeddings[select_index], color))

    return scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate sequence embeddings on CATH (aligned to original evaluate_with_cath_more).")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--cath_seq", type=str, required=True, help="CATH fasta (headers like 1.10.10.2080|cath|...)")

    parser.add_argument("--truncate_inference", default=None, type=int, choices=[0, 1])
    parser.add_argument("--max_length_inference", default=None, type=int, nargs="?")
    parser.add_argument("--afterproject", action="store_true")
    parser.add_argument("--steps", type=int, default=0)
    args = parser.parse_args()

    base = os.path.splitext(args.config_path)[0]
    out_figure_path = os.path.join(base, "CATH_test_release_seq")
    Path(out_figure_path).mkdir(parents=True, exist_ok=True)

    scores = evaluate_with_cath_more_seq_aligned(
        out_figure_path=out_figure_path,
        steps=args.steps,
        cath_fasta_path=args.cath_seq,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        truncate_inference=args.truncate_inference,
        max_length_inference=args.max_length_inference,
        afterproject=args.afterproject,
    )

    (
        ch1_full, ch1_tsne, ari1, sil1,
        ch2_full, ch2_tsne, ari2, sil2,
        ch3_full, ch3_tsne, ari3, sil3,
    ) = scores

    print("\n=== CATH clustering evaluation (sequence embeddings) ===")
    print(
        f"[Calinski–Harabasz score]\n"
        f"  - Class level (digit 1):       full = {ch1_full:.4f}, t-SNE = {ch1_tsne:.4f}\n"
        f"  - Architecture level (digit 2): full = {ch2_full:.4f}, t-SNE = {ch2_tsne:.4f}\n"
        f"  - Fold level (digit 3):         full = {ch3_full:.4f}, t-SNE = {ch3_tsne:.4f}"
    )
    print(
        f"\n[Adjusted Rand index (ARI) for k-means on t-SNE embeddings]\n"
        f"  - Class level (digit 1):       {ari1:.4f}\n"
        f"  - Architecture level (digit 2): {ari2:.4f}\n"
        f"  - Fold level (digit 3):         {ari3:.4f}"
    )
    print(
        f"\n[Silhouette score (using full-dimensional embeddings)]\n"
        f"  - Class level (digit 1):       {sil1:.4f}\n"
        f"  - Architecture level (digit 2): {sil2:.4f}\n"
        f"  - Fold level (digit 3):         {sil3:.4f}\n"
    )

    scores_file = os.path.join(out_figure_path, "scores.txt")
    with open(scores_file, "w") as f:
        f.write("CATH clustering evaluation (sequence embeddings) - aligned to original evaluate_with_cath_more\n")
        f.write("========================================================================================\n\n")

        f.write("[Calinski–Harabasz score]\n")
        f.write(
            f"  - Class level (digit 1):       full = {ch1_full:.4f}, t-SNE = {ch1_tsne:.4f}\n"
            f"  - Architecture level (digit 2): full = {ch2_full:.4f}, t-SNE = {ch2_tsne:.4f}\n"
            f"  - Fold level (digit 3):         full = {ch3_full:.4f}, t-SNE = {ch3_tsne:.4f}\n\n"
        )

        f.write("[Adjusted Rand index (ARI) for k-means on t-SNE embeddings]\n")
        f.write(
            f"  - Class level (digit 1):       {ari1:.4f}\n"
            f"  - Architecture level (digit 2): {ari2:.4f}\n"
            f"  - Fold level (digit 3):         {ari3:.4f}\n\n"
        )

        f.write("[Silhouette score (using full-dimensional embeddings)]\n")
        f.write(
            f"  - Class level (digit 1):       {sil1:.4f}\n"
            f"  - Architecture level (digit 2): {sil2:.4f}\n"
            f"  - Fold level (digit 3):         {sil3:.4f}\n"
        )

    print(f"Scores written to {scores_file}")
    print(f"Figures saved to {out_figure_path}")


if __name__ == "__main__":
    main()
