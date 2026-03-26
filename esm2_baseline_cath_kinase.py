"""
CATH / Kinase clustering eval using **plain fair-esm ESM2** (no S-PLM checkpoint).

Same protocol as cath_with_seq.py and kinase_with_seq.py (t-SNE, CH, ARI, Silhouette,
output figures and scores.txt), but embeddings come from
`esm.pretrained.esm2_t33_650M_UR50D()` with mean pooling over residue tokens.

Requires: ``fair-esm`` (see ``install.sh`` / ``environment.yaml`` in this repo).

This script is **standalone**: logic matches SPLM-V2-GVP's ``esm2_baseline_cath_kinase.py``;
only repo-specific paths in the docstring differ.

Always runs **both** CATH and Kinase. Outputs go under ``{result_path}/cath/`` and
``{result_path}/kinase/`` (default ``--result_path`` is ``./``).

Example
-------
  cd /path/to/S-PLM
  python esm2_baseline_cath_kinase.py \\
    --cath_seq ./dataset/Rep_subfamily_basedon_S40pdb.fa \\
    --kinase_seq ./dataset/kinase_alllabels.fa

  python esm2_baseline_cath_kinase.py \\
    --cath_seq ... --kinase_seq ... --result_path ./out_esm2/
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from Bio import SeqIO

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score, adjusted_rand_score

import esm


def load_esm2_t33():
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, alphabet, device


def embed_sequences_esm2(
    seqs: list[str],
    model,
    alphabet,
    device: torch.device,
    truncation_seq_length: int | None = None,
    batch_size: int = 4,
) -> np.ndarray:
    """Return array [N, D] in the same order as seqs."""
    if truncation_seq_length is not None:
        batch_converter = alphabet.get_batch_converter(truncation_seq_length=truncation_seq_length)
    else:
        batch_converter = alphabet.get_batch_converter()

    out_list = []
    n = len(seqs)
    for start in range(0, n, batch_size):
        chunk = seqs[start : start + batch_size]
        batch = [(str(start + i), s) for i, s in enumerate(chunk)]
        _, _, batch_tokens = batch_converter(batch)
        batch_tokens = batch_tokens.to(device)
        with torch.inference_mode():
            results = model(
                batch_tokens,
                repr_layers=[model.num_layers],
                return_contacts=False,
            )
            x = results["representations"][model.num_layers]

        mask = (
            (batch_tokens != alphabet.padding_idx)
            & (batch_tokens != alphabet.cls_idx)
            & (batch_tokens != alphabet.eos_idx)
        ).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1)
        protein_vec = (x * mask).sum(dim=1) / denom
        out_list.append(protein_vec.float().cpu().numpy())

    return np.concatenate(out_list, axis=0)


def read_cath_fasta_like_original(cath_fasta_path: str):
    labels = []
    seqs = []
    with open(cath_fasta_path, "r") as f:
        for line in f:
            if not line.startswith(">"):
                continue
            labels.append(line[1:].split("|")[0].strip())
            seq_line = next(f)
            seqs.append(seq_line.strip())
    return labels, seqs


def scatter_labeled_z_cath(
    z_batch,
    colors,
    filename="test_plot.png",
    legend_labels=None,
    legend_title=None,
):
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
                    [0],
                    [0],
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


def run_cath_esm2(
    out_figure_path: str,
    cath_fasta_path: str,
    truncate_inference: int | None,
    max_length_inference: int | None,
    batch_size: int,
):
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
    labels, seqs = read_cath_fasta_like_original(cath_fasta_path)

    model, alphabet, device = load_esm2_t33()
    trunc = None
    if truncate_inference is not None and int(truncate_inference) == 1:
        assert max_length_inference is not None
        trunc = int(max_length_inference)

    seq_embeddings = embed_sequences_esm2(
        seqs, model, alphabet, device, truncation_seq_length=trunc, batch_size=batch_size
    )

    n_show = min(50, len(seq_embeddings))
    if n_show > 0:
        distances = torch.cdist(
            torch.tensor(seq_embeddings[:n_show, :], dtype=torch.float64),
            torch.tensor(seq_embeddings[:n_show, :], dtype=torch.float64),
        )
        print(f"cath Min distance: {distances.min()}")
        print(f"cath Max distance: {distances.max()}")
        print(f"cath Mean distance: {distances.mean()}")
        print(f"cath Std distance: {distances.std()}")

    mdel = TSNE(n_components=2, random_state=0, init="random", method="exact")
    z_tsne_seq = mdel.fit_transform(seq_embeddings)

    scores = []
    tol_archi_seq = {"3.30", "3.40", "1.10", "3.10", "2.60"}
    tol_fold_seq = {"1.10.10", "3.30.70", "2.60.40", "2.60.120", "3.40.50"}

    for digit_num in [1, 2, 3]:
        color = []
        colorid = []
        keys = {}
        colorindex = 0

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

            color.append(ct[(keys[key]) % len(ct)])
            colorid.append(keys[key])
            select_index.append(labels.index(label))
            color_dict[keys[key]] = ct[keys[key]]

        print(f"sample num (digit {digit_num}) = {len(select_index)}")
        if len(select_index) == 0:
            scores.extend([float("nan")] * 4)
            continue

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

        scores.append(calinski_harabasz_score(seq_embeddings[select_index], color))
        scores.append(calinski_harabasz_score(z_tsne_seq[select_index], color))

        scatter_labeled_z_cath(
            z_tsne_seq[select_index],
            color,
            filename=os.path.join(out_figure_path, f"CATH_seqrep_{digit_num}.png"),
            legend_labels=legend_labels,
            legend_title=legend_title,
        )

        kmeans = KMeans(n_clusters=len(color_dict), random_state=42)
        predicted_labels = kmeans.fit_predict(z_tsne_seq[select_index])
        ari = adjusted_rand_score(colorid, predicted_labels)
        scores.append(ari)
        scores.append(silhouette_score(seq_embeddings[select_index], color))

    return scores, "CATH (fair-esm ESM2 t33 650M UR50D baseline)"


def write_cath_scores_txt(result_path: str, scores: list, title: str):
    (
        ch1_full, ch1_tsne, ari1, sil1,
        ch2_full, ch2_tsne, ari2, sil2,
        ch3_full, ch3_tsne, ari3, sil3,
    ) = scores
    scores_file = os.path.join(result_path, "scores.txt")
    with open(scores_file, "w") as f:
        f.write(f"{title}\n")
        f.write("=" * len(title) + "\n\n")
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


def print_cath_summary(title: str, scores: list):
    (
        ch1_full, ch1_tsne, ari1, sil1,
        ch2_full, ch2_tsne, ari2, sil2,
        ch3_full, ch3_tsne, ari3, sil3,
    ) = scores
    print(f"\n=== {title} ===")
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


def read_kinase_fasta_allinfo(fn_fasta: str):
    rows = []
    with open(fn_fasta, "r") as handle:
        for rec in SeqIO.parse(handle, "fasta"):
            header = rec.description
            parts = header.split("|")
            meta = {}
            for token in parts[1:]:
                if "=" in token:
                    k, v = token.split("=", 1)
                    meta[k.strip()] = v.strip().replace("%7C", "|")
            rows.append({
                "Kinase_Entrez_Symbol": meta.get("Kinase_Entrez_Symbol", ""),
                "Kinase_group": meta.get("Kinase_group", ""),
                "Kinase_seq": str(rec.seq),
            })
    return rows


def scatter_labeled_z_kinase(
    z_batch,
    colors,
    filename="test_plot.png",
    legend_labels=None,
    legend_title=None,
):
    plt.switch_backend("Agg")
    fig = plt.gcf()
    fig.set_size_inches(4, 4)
    plt.clf()

    for n in range(z_batch.shape[0]):
        plt.scatter(
            z_batch[n, 0],
            z_batch[n, 1],
            c=[colors[n]] if isinstance(colors[n], tuple) else colors[n],
            s=50,
            marker="o",
            edgecolors="none",
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
                    [0],
                    [0],
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


def run_kinase_esm2(out_figure_path: str, kinase_seq: str, batch_size: int):
    Path(out_figure_path).mkdir(parents=True, exist_ok=True)
    rows = read_kinase_fasta_allinfo(kinase_seq)

    seen = set()
    dedup_rows = []
    for r in rows:
        sym = r["Kinase_Entrez_Symbol"]
        if sym in seen:
            continue
        seen.add(sym)
        dedup_rows.append(r)

    labels = []
    seqs = []
    for r in dedup_rows:
        if r["Kinase_seq"] == " ":
            continue
        labels.append(r["Kinase_group"])
        seqs.append(r["Kinase_seq"])

    model, alphabet, device = load_esm2_t33()
    seq_embeddings = embed_sequences_esm2(seqs, model, alphabet, device, batch_size=batch_size)

    n_show = min(50, len(seq_embeddings))
    if n_show > 0:
        distances = torch.cdist(
            torch.tensor(seq_embeddings[:n_show, :], dtype=torch.float64),
            torch.tensor(seq_embeddings[:n_show, :], dtype=torch.float64),
        )
        print(f"Kinase Min distance: {distances.min()}")
        print(f"Kinase Max distance: {distances.max()}")
        print(f"Kinase Mean distance: {distances.mean()}")
        print(f"Kinase Std distance: {distances.std()}")

    mdel = TSNE(n_components=2, random_state=0, init="random", method="exact")
    z_tsne_seq = mdel.fit_transform(seq_embeddings)

    ct = {
        "AGC": (154 / 255, 99 / 255, 36 / 255),
        "Atypical": (253 / 255, 203 / 255, 110 / 255),
        "Other": (58 / 255, 179 / 255, 73 / 255),
        "STE": (255 / 255, 114 / 255, 114 / 255),
        "TK": (63 / 255, 96 / 255, 215 / 255),
        "CK1": (151 / 255, 130 / 255, 255 / 255),
        "CMGC": (141 / 255, 23 / 255, 178 / 255),
        "TKL": (62 / 255, 211 / 255, 244 / 255),
        "CAMK": (190 / 255, 239 / 255, 65 / 255),
    }

    scores = []
    color = []
    colorid = []
    keys = {}
    colorindex = 0
    color_dict = {}

    if len(labels) != len(seq_embeddings):
        print(
            f"[WARN] labels ({len(labels)}) != embeddings ({len(seq_embeddings)}). "
            f"Original code assumes they match."
        )

    for label in labels[: len(seq_embeddings)]:
        key = label
        if key not in keys:
            keys[key] = colorindex
            colorindex += 1
        colorid.append(keys[key])
        color.append(ct[key])
        color_dict[keys[key]] = ct[key]

    legend_labels = {group_name: ct[group_name] for group_name, gid in keys.items()}

    scatter_labeled_z_kinase(
        z_tsne_seq,
        color,
        filename=os.path.join(out_figure_path, "kinase_group_seqrep.png"),
        legend_labels=legend_labels,
        legend_title="Kinase group",
    )

    scores.append(calinski_harabasz_score(seq_embeddings, colorid[: len(seq_embeddings)]))
    scores.append(calinski_harabasz_score(z_tsne_seq, colorid[: len(seq_embeddings)]))
    kmeans = KMeans(n_clusters=len(color_dict), random_state=42)
    predicted_labels = kmeans.fit_predict(z_tsne_seq)
    ari = adjusted_rand_score(colorid[: len(seq_embeddings)], predicted_labels)
    scores.append(ari)
    scores.append(silhouette_score(seq_embeddings, colorid[: len(seq_embeddings)]))

    return scores, "Kinase (fair-esm ESM2 t33 650M UR50D baseline)"


def write_kinase_scores_txt(result_path: str, scores: list, title: str):
    ch_full, ch_tsne, ari, sil = scores
    scores_file = os.path.join(result_path, "scores.txt")
    with open(scores_file, "w") as f:
        f.write(f"{title}\n")
        f.write("=" * len(title) + "\n\n")
        f.write("[Calinski–Harabasz score]\n")
        f.write(f"  - full = {ch_full:.4f}, t-SNE = {ch_tsne:.4f}\n\n")
        f.write("[Adjusted Rand index (ARI) for k-means on t-SNE embeddings]\n")
        f.write(f"  - {ari:.4f}\n\n")
        f.write("[Silhouette score (using full-dimensional embeddings)]\n")
        f.write(f"  - {sil:.4f}\n")
    print(f"Scores written to {scores_file}")


def print_kinase_summary(title: str, scores: list):
    ch_full, ch_tsne, ari, sil = scores
    print(f"\n=== {title} ===")
    print(f"[Calinski–Harabasz score]\n  - full = {ch_full:.4f}, t-SNE = {ch_tsne:.4f}")
    print(f"\n[Adjusted Rand index (ARI) for k-means on t-SNE embeddings]\n  - {ari:.4f}")
    print(f"\n[Silhouette score (using full-dimensional embeddings)]\n  - {sil:.4f}\n")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run CATH then Kinase with plain fair-esm ESM2 t33 650M UR50D (no S-PLM checkpoint). "
            "Writes result_path/cath/ and result_path/kinase/ (default result_path: ./)."
        )
    )
    parser.add_argument("--cath_seq", type=str, required=True, help="CATH FASTA path")
    parser.add_argument("--kinase_seq", type=str, required=True, help="Kinase FASTA path")
    parser.add_argument(
        "--result_path",
        type=str,
        default="./",
        help="Output root directory (default: ./). Subdirs cath/ and kinase/ are created.",
    )
    parser.add_argument("--truncate_inference", type=int, default=None, choices=[0, 1])
    parser.add_argument("--max_length_inference", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=4)

    args = parser.parse_args()

    base = args.result_path
    Path(base).mkdir(parents=True, exist_ok=True)
    cath_dir = os.path.join(base, "cath")
    kinase_dir = os.path.join(base, "kinase")

    print("=== Running CATH (ESM2 baseline) ===")
    cath_scores, cath_title = run_cath_esm2(
        cath_dir,
        args.cath_seq,
        args.truncate_inference,
        args.max_length_inference,
        args.batch_size,
    )
    print_cath_summary(cath_title, cath_scores)
    write_cath_scores_txt(cath_dir, cath_scores, cath_title)

    print("\n=== Running Kinase (ESM2 baseline) ===")
    kin_scores, kin_title = run_kinase_esm2(kinase_dir, args.kinase_seq, args.batch_size)
    print_kinase_summary(kin_title, kin_scores)
    write_kinase_scores_txt(kinase_dir, kin_scores, kin_title)
    print(f"Figure saved to {os.path.join(kinase_dir, 'kinase_group_seqrep.png')}")
    print(f"\nOutputs: {os.path.abspath(cath_dir)}/  and  {os.path.abspath(kinase_dir)}/")


if __name__ == "__main__":
    main()
