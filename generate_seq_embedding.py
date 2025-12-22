# generate_seq_embedding.py
import argparse
import os
import pickle
import yaml
from pathlib import Path
from typing import Dict, Tuple

from Bio import SeqIO
import torch
import numpy as np

from utils import load_configs, load_checkpoints_only
from model import SequenceRepresentation


def read_fasta(fn_fasta: str) -> Dict[str, str]:
    prot2seq = {}
    with open(fn_fasta) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            prot2seq[record.id] = str(record.seq)
    return prot2seq


def build_model_and_converter(config_path: str,
                              checkpoint_path: str = None,
                              truncate_inference: str = None,
                              max_length_inference: int = None
                              ) -> Tuple[SequenceRepresentation, callable, torch.device]:
    # Load configuration
    with open(config_path) as f:
        dict_config = yaml.full_load(f)
    configs = load_configs(dict_config)

    # Build the model
    model = SequenceRepresentation(logging=None, configs=configs)
    model.eval()

    # Load checkpoint
    if checkpoint_path is not None and len(checkpoint_path) > 0:
        load_checkpoints_only(checkpoint_path, model)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Batch converter (compatible with truncation)
    if hasattr(model, "alphabet") and model.alphabet is not None:
        if truncate_inference is not None:
            # Allow values like "1"/"0" or True/False
            do_trunc = str(truncate_inference).lower() in ["1", "true", "yes"]
            if do_trunc:
                assert max_length_inference is not None, \
                    "truncate_inference is on, you must provide max_length_inference"
                batch_converter = model.alphabet.get_batch_converter(
                    truncation_seq_length=int(max_length_inference)
                )
                print(f"truncate_inference with max_length={max_length_inference}")
            else:
                batch_converter = model.alphabet.get_batch_converter()
        else:
            # Use default if not overridden
            batch_converter = model.alphabet.get_batch_converter()
    else:
        # Fallback: use the attribute from the first version
        batch_converter = model.batch_converter

    return model, batch_converter, device


def embed_one_sequence(model: SequenceRepresentation,
                 batch_converter,
                 device: torch.device,
                 prot_id: str,
                 seq: str,
                 residue_level: bool):
    # Assemble data as required by ESM: a list of (name, seq)
    esm2_seq = [(prot_id, seq)]
    with torch.inference_mode():
        _, _, batch_tokens = batch_converter(esm2_seq)
        batch_tokens = batch_tokens.to(device)

        protein_representation, residue_representation, mask = model(batch_tokens)

        if residue_level:
            out = residue_representation
        else:
            out = protein_representation

        out = out.detach().float().cpu().numpy()

        # ---- Normalize shapes ----
        if not residue_level:
            return out

        else:
            # residue-level:
            return out.squeeze(0)


def generate_seq_embedding(
    query_sequence: Dict[str, str],
    config_path: str,
    checkpoint_path: str = None,
    truncate_inference=None,
    max_length_inference: int = None,
    afterproject: bool = False,
    residue_level: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Generate embeddings for a dict of sequences.

    Parameters
    ----------
    query_sequence : Dict[str, str]
        {protein_id -> amino_acid_sequence}
    config_path : str
        YAML config path.
    checkpoint_path : str, optional
        Model checkpoint (.pth).
    truncate_inference : optional
        If truthy ("1"/1/True), enable truncation.
    max_length_inference : int, optional
        Max sequence length used when truncation is enabled.
    afterproject : bool
        Keep for API compatibility. If your model exposes a post-projection output,
        you should implement it inside `embed_one_sequence` or inside the model forward.
    residue_level : bool
        If True, return per-residue embeddings; else return per-protein embeddings.

    Returns
    -------
    Dict[str, np.ndarray]
        {protein_id -> embedding}
        - residue_level=False: embedding shape (D,)
        - residue_level=True : embedding shape (L, D)
    """
    model, batch_converter, device = build_model_and_converter(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        truncate_inference=truncate_inference,
        max_length_inference=max_length_inference,
    )

    result = {}
    for pid, seq in query_sequence.items():
        result[pid] = embed_one_sequence(
            model=model,
            batch_converter=batch_converter,
            device=device,
            prot_id=pid,
            seq=seq,
            residue_level=residue_level,
            afterproject=afterproject,
        )

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequence embedding CLI using SequenceRepresentation backend")
    parser.add_argument("--input_seq", required=True, help="Path to FASTA file")
    parser.add_argument("--result_path", required=True, help="Output directory")
    parser.add_argument("--out_file", default="protein_embeddings.pkl", help="Output file name, default protein_embeddings.pkl")
    parser.add_argument("--config_path", "-c", required=True, help="Configuration file path representation_config.yaml")
    parser.add_argument("--checkpoint_path", default=None, help="Optional S-PLM checkpoint path")
    parser.add_argument("--truncate_inference", default=0, help="1 or 0. If 1, must also provide --max_length_inference")
    parser.add_argument("--max_length_inference", type=int, default=None, help="Truncation length")
    parser.add_argument("--residue_level", action="store_true", help="Output residue-level representations")
    parser.add_argument("--afterproject", action="store_true", help="Output post-projection representations if supported")
    args = parser.parse_args()

    Path(args.result_path).mkdir(parents=True, exist_ok=True)
    save_path = os.path.join(args.result_path, args.out_file)

    # Read FASTA (id -> sequence)
    prot2seq = read_fasta(args.input_seq)

    # Generate embeddings using the unified function API
    result = generate_seq_embedding(
        query_sequence=prot2seq,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        truncate_inference=args.truncate_inference,
        max_length_inference=args.max_length_inference,
        afterproject=args.afterproject,
        residue_level=args.residue_level,
    )

    # Save
    with open(save_path, "wb") as f:
        pickle.dump(result, f)

    print(f"Saved embeddings to {save_path}")


"""
1. Basic usage

Generate embeddings for sequences in test.fasta using your config and checkpoint:

python generate_seq_embedding.py \
  --input_seq ./test.fasta \
  --config_path ./configs/representation_config.yaml \
  --checkpoint_path /path/to/checkpoint.pth \
  --result_path ./out

2. With sequence truncation

Limit the maximum sequence length to 1022 residues during inference:

python generate_seq_embedding.py \
  --input_seq ./test.fasta \
  --config_path ./configs/representation_config.yaml \
  --checkpoint_path /path/to/checkpoint.pth \
  --result_path ./out \
  --out_file truncate_protein_embeddings.pkl \
  --truncate_inference 1 \
  --max_length_inference 1022

3. Residue-level embeddings

Output per-residue embeddings (instead of per-protein representations):

python generate_seq_embedding.py \
  --input_seq ./test.fasta \
  --config_path ./configs/representation_config.yaml \
  --checkpoint_path /path/to/checkpoint.pth \
  --result_path ./out \
  --residue_level

import pickle
with open('./out/protein_embeddings.pkl', 'rb') as f:
    data = pickle.load(f)


python generate_seq_embedding.py \
  --input_seq "/cluster/pixstor/xudong-lab/duolin/splm_gvpgit/PLMsearch/plmsearch_data/swissprot_to_swissprot_5/protein.fasta" \
  --config_path ./configs/representation_config.yaml \
  --checkpoint_path "/cluster/pixstor/xudong-lab/duolin/splm_gvpgit/results/checkpoints_configs/checkpoint_0280000.pth" \
  --result_path ./out \
  --residue_level

"""


