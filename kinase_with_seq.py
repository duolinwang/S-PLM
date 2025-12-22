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

from utils.generate_seq_embedding import generate_seq_embedding
from Bio import SeqIO


def read_kinase_fasta_allinfo(fn_fasta: str):
    """
    Read the 'all-columns-in-header' kinase FASTA.

    Header example:
      >KIN_00000001|Kinase_Entrez_Symbol=AAK1|Kinase_group=Other|...|Substrate_code=T
      SEQUENCE (Kinase_seq)

    Returns
    -------
    rows : list[dict]
      Each row has keys:
        - Kinase_Entrez_Symbol
        - Kinase_group
        - Kinase_seq
      (Other columns are ignored by the ORIGINAL evaluation protocol.)
    """
    rows = []
    with open(fn_fasta, "r") as handle:
        for rec in SeqIO.parse(handle, "fasta"):
            header = rec.description  # full header line without leading '>'
            parts = header.split("|")

            meta = {}
            for token in parts[1:]:
                if "=" in token:
                    k, v = token.split("=", 1)
                    # undo minimal escape if you used it when writing fasta
                    meta[k.strip()] = v.strip().replace("%7C", "|")

            rows.append({
                "Kinase_Entrez_Symbol": meta.get("Kinase_Entrez_Symbol", ""),
                "Kinase_group": meta.get("Kinase_group", ""),
                "Kinase_seq": str(rec.seq),
            })
    return rows


def scatter_labeled_z(
    z_batch,
    colors,
    filename="test_plot.png",
    legend_labels=None,
    legend_title=None
):
    """
    Save a 2D scatter plot colored by class.
    Style aligned to your existing cath_with_struct.py.

    Parameters
    ----------
    z_batch : np.ndarray, shape [N, 2]
        2D embeddings (e.g., t-SNE output).
    colors : list
        Per-point colors. Here we use RGB tuples from the original kinase eval.
    filename : str
        Output path for the PNG file.
    legend_labels : dict, optional
        {label_str: color} mapping for legend.
    legend_title : str, optional
        Legend title.
    """
    plt.switch_backend("Agg")
    fig = plt.gcf()
    fig.set_size_inches(4, 4)
    plt.clf()

    for n in range(z_batch.shape[0]):
        plt.scatter(
            z_batch[n, 0],
            z_batch[n, 1],
            # matplotlib expects list of tuples for RGB
            c=[colors[n]] if isinstance(colors[n], tuple) else colors[n],
            s=50,
            marker="o",
            edgecolors="none"
        )

    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
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


def evaluate_with_kinase_original_protocol_using_generate(
    out_figure_path,
    kinase_seq,
    config_path,
    checkpoint_path,
):
    """
    EXACTLY replicate the original evaluate_with_kinase protocol,
    but generate embeddings via utils.generate_seq_embedding.generate_seq_embedding.

    Original protocol summary:
      1) Read TSV with pandas (sep='\\t')
      2) Drop duplicates by Kinase_Entrez_Symbol (keep first)
      3) Filter rows by: Kinase_seq != ' '  (a single space)
      4) labels = Kinase_group; sequences = Kinase_seq
      5) Compute sequence embeddings (features_seq branch, i.e. afterproject=False)
      6) t-SNE (init='random', method='exact', random_state=0)
      7) Color mapping: fixed Kinase_group -> RGB tuple dict (ct)
      8) Metrics:
         - Calinski-Harabasz on full embeddings (with integer class ids)
         - Calinski-Harabasz on t-SNE 2D embeddings
         - KMeans on t-SNE (k = num classes) + ARI against true class ids
         - Silhouette on full embeddings
      9) Save figure as: kinase_group_seqrep.png


    Returns
    -------
    scores : list[float]
        [CH_full, CH_tsne, ARI, silhouette]
    """
    Path(out_figure_path).mkdir(parents=True, exist_ok=True)

    # --- Minimal change: read FASTA instead of TSV, but keep EXACT original semantics ---
    rows = read_kinase_fasta_allinfo(kinase_seq)

    # drop_duplicates(subset="Kinase_Entrez_Symbol", keep="first")
    seen = set()
    dedup_rows = []
    for r in rows:
        sym = r["Kinase_Entrez_Symbol"]
        if sym in seen:
            continue
        seen.add(sym)
        dedup_rows.append(r)

    # filter: Kinase_seq != " " (a single space), EXACT same check
    labels = []
    seqs = []
    for r in dedup_rows:
        if r["Kinase_seq"] == " ":
            continue
        labels.append(r["Kinase_group"])
        seqs.append(r["Kinase_seq"])

    # --- MUST use generate_seq_embedding: build query_sequence ---
    # NOTE: dict preserves insertion order in Python 3.7+, so the embedding order matches seqs order.
    query_sequence = {str(i): str(seqs[i]) for i in range(len(seqs))}

    query_embedding_dic = generate_seq_embedding(
        query_sequence=query_sequence,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        truncate_inference=None,
        max_length_inference=None,
        afterproject=False,   # original uses features_seq (not projected)
        residue_level=False,
    )

    # --- Build embeddings matrix (N, D) in insertion order ---
    vals = list(query_embedding_dic.values())
    seq_embeddings = np.concatenate(vals, axis=0)
    seq_embeddings = np.asarray(seq_embeddings)

    # --- Original: print pairwise distance stats for first 50 samples ---
    distances = torch.cdist(
        torch.tensor(seq_embeddings[:50, :], dtype=torch.float64),
        torch.tensor(seq_embeddings[:50, :], dtype=torch.float64)
    )
    print(f"Kinase Min distance: {distances.min()}")
    print(f"Kinase Max distance: {distances.max()}")
    print(f"Kinase Mean distance: {distances.mean()}")
    print(f"Kinase Std distance: {distances.std()}")

    # --- Original: t-SNE projection ---
    mdel = TSNE(n_components=2, random_state=0, init="random", method="exact")
    z_tsne_seq = mdel.fit_transform(seq_embeddings)

    # --- Original: fixed Kinase_group -> RGB mapping ---
    ct = {
        "AGC": (154 / 255, 99 / 255, 36 / 255),
        "Atypical": (253 / 255, 203 / 255, 110 / 255),
        "Other": (58 / 255, 179 / 255, 73 / 255),
        "STE": (255 / 255, 114 / 255, 114 / 255),
        "TK": (63 / 255, 96 / 255, 215 / 255),
        "CK1": (151 / 255, 130 / 255, 255 / 255),  # note!
        "CMGC": (141 / 255, 23 / 255, 178 / 255),
        "TKL": (62 / 255, 211 / 255, 244 / 255),
        "CAMK": (190 / 255, 239 / 255, 65 / 255),
    }

    # --- Build integer class ids and colors (same logic as original) ---
    scores = []
    color = []
    colorid = []
    keys = {}
    colorindex = 0
    color_dict = {}

    # The original code assumes len(labels) == len(embeddings)
    # If generate_seq_embedding silently drops some sequences, this mismatch may happen.
    if len(labels) != len(seq_embeddings):
        print(
            f"[WARN] labels ({len(labels)}) != embeddings ({len(seq_embeddings)}). "
            f"Original code assumes they match."
        )

    for label in labels[:len(seq_embeddings)]:
        key = label
        if key not in keys:
            keys[key] = colorindex
            colorindex += 1

        colorid.append(keys[key])
        color.append(ct[key])          # original will KeyError if key not in ct
        color_dict[keys[key]] = ct[key]

    # --- save the t-SNE scatter plot ---
    legend_labels = {}
    for group_name, gid in keys.items():
        legend_labels[group_name] = ct[group_name]

    scatter_labeled_z(
        z_tsne_seq,
        color,
        filename=os.path.join(out_figure_path, f"kinase_group_seqrep.png"),
        legend_labels=legend_labels,
        legend_title="Kinase group",
    )

    # --- Original: clustering metrics ---
    scores.append(calinski_harabasz_score(seq_embeddings, colorid[:len(seq_embeddings)]))
    scores.append(calinski_harabasz_score(z_tsne_seq, colorid[:len(seq_embeddings)]))

    kmeans = KMeans(n_clusters=len(color_dict), random_state=42)
    predicted_labels = kmeans.fit_predict(z_tsne_seq)

    ari = adjusted_rand_score(colorid[:len(seq_embeddings)], predicted_labels)
    scores.append(ari)

    scores.append(silhouette_score(seq_embeddings, colorid[:len(seq_embeddings)]))

    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Kinase seq clustering eval (ORIGINAL protocol, using generate_seq_embedding)."
    )
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--kinase_seq", type=str, required=True)
    parser.add_argument("--result_path", type=str, default="./")
    args = parser.parse_args()

    out_figure_path = args.result_path
    Path(out_figure_path).mkdir(parents=True, exist_ok=True)

    scores = evaluate_with_kinase_original_protocol_using_generate(
        out_figure_path=out_figure_path,
        kinase_seq=args.kinase_seq,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
    )

    ch_full, ch_tsne, ari, sil = scores

    print("\n=== Kinase clustering evaluation (sequence embeddings) ===")
    print(
        f"[Calinski–Harabasz score]\n"
        f"  - full = {ch_full:.4f}, t-SNE = {ch_tsne:.4f}"
    )
    print(
        f"\n[Adjusted Rand index (ARI) for k-means on t-SNE embeddings]\n"
        f"  - {ari:.4f}"
    )
    print(
        f"\n[Silhouette score (using full-dimensional embeddings)]\n"
        f"  - {sil:.4f}\n"
    )

    scores_file = os.path.join(out_figure_path, "scores.txt")
    with open(scores_file, "w") as f:
        f.write("Kinase clustering evaluation (sequence embeddings) - ORIGINAL PROTOCOL\n")
        f.write("====================================================================\n\n")

        f.write("[Calinski–Harabasz score]\n")
        f.write(f"  - full = {ch_full:.4f}, t-SNE = {ch_tsne:.4f}\n\n")

        f.write("[Adjusted Rand index (ARI) for k-means on t-SNE embeddings]\n")
        f.write(f"  - {ari:.4f}\n\n")

        f.write("[Silhouette score (using full-dimensional embeddings)]\n")
        f.write(f"  - {sil:.4f}\n")

    print(f"Scores written to {scores_file}")
    print(f"Figure saved to {os.path.join(out_figure_path, f'kinase_group_seqrep.png')}")


if __name__ == "__main__":
    main()
