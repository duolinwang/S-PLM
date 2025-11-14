# S-PLM: Structure-aware Protein Language Model via Contrastive Learning between Sequence and Structure

<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Oryx Video-ChatGPT">
</p>

[![paper](https://img.shields.io/badge/bioRxiv-Paper-<COLOR>.svg)](https://www.biorxiv.org/content/10.1101/2023.08.06.552203v3)

## S-PLM V1
This is the official implementation of S-PLM paper (S-PLM V1). S-PLM is a 3D structure-aware protein language model (PLM) that enables the sequence-based embedding to carry the structural information through multi-view contrastive learning. 

This repository offers comprehensive guidance on utilizing pre-trained S-PLM models to generate structure-aware protein representations. Additionally, it provides a library of code for implementing lightweight tuning methods tailored for various downstream supervised learning tasks involving proteins.

The tasks include Enzyme Commission number (EC) prediction, Gene Ontology (GO) prediction, protein fold (fold) and enzyme reaction (ER) classification, and protein secondary structure (SS) prediction. 

The lightweight tuning methods include fine-tune top layers, Adapter Tuning, and Low-rank adaptation (LoRA). Users can train a task-specific model using different tuning methods by modifying the configuration files provided in the [configs directory](https://github.com/duolinwang/S-PLM/tree/main/configs)
## S-PLM V2
To use an updated residue-level pre-trained model, refer to [model explanation](https://github.com/duolinwang/S-PLM/blob/main/model/Model%20readme.md)
> We now provide a new GVP-based structure encoder. See [SPLM-V2-GVP](https://github.com/Yichuan0712/SPLM-V2-GVP)
.
## Installation
To use S-PLM project, install the corresponding environment.yaml file in your environment. Or you can follow the install.sh file to install the dependencies.

### Install using yaml file
Using `environment.yaml`
1. Create a new environment using the `environment.yaml` file: `conda env create -f environment.yaml`.
2. Activate the environment you have just created: `conda activate splm`.

### Install using SH file
Create a conda environment and use this command to install the required packages inside the conda environment.
First, make the install.sh file executable by running the following command:
```commandline
chmod +x install.sh
```
Then, run the following command to install the required packages inside the conda environment:
```commandline
bash install.sh
```

## Run
[![Colab quickstart](https://img.shields.io/badge/Colab-quickstart-e91e63)](#)
>**Colab quickstart available:** Use our minimal [**S-PLM v1 Colab example**](https://github.com/duolinwang/S-PLM/blob/main/SPLM_v1_sequence_embedding_quickstart.ipynb) to set up dependencies, load the checkpoint, extract embeddings, and launch downstream training.



### Use S-PLM for downstream tasks
To utilize the accelerator power in your training code, such as distributed multi-GPU training, you have to set the accelerator config by running `accelerate config` in the command line.
Then, you have to set the training settings and hyperparameters inside your target task `configs/config_{task}.yaml` file.
Finally, you can start your training for downstream tasks using a config file from configs and a pretrained [S-PLM model](https://mailmissouri-my.sharepoint.com/:f:/g/personal/wangdu_umsystem_edu/Evk7BBT5LxRMpsHzKxmi0DEBrgv1mgBK0MRuRHJSqSoHZQ?e=Eozrwh) by running
```commandline
accelerate launch train_{task}.py --config_path configs/<config_name> --resume_path model/checkpoint_0520000.pth`
```

### 🚀 Examples 
#### Training and evaluation of GO prediction tasks
```commandline
accelerate launch train_go.py --config_path configs/bp_config_adapterH_adapterH.yaml --resume_path model/checkpoint_0520000.pth
accelerate launch train_go.py --config_path configs/cc_config_adapterH_adapterH.yaml --resume_path model/checkpoint_0520000.pth
accelerate launch train_go.py --config_path configs/mf_config_adapterH_adapterH.yaml --resume_path model/checkpoint_0520000.pth
```
#### Train and evaluate fold classification
```commandline
accelerate launch train_fold.py --config_path configs/fold_config_adapterH_finetune.yaml --resume_path model/checkpoint_0520000.pth
```
#### Train and evaluate of secondary structure prediction
```commandline
accelerate launch train_ss.py --config_path configs/ss_config_adapterH_finetune.yaml --resume_path model/checkpoint_0520000.pth
```
You might not use accelerator to run the `train.py` script if you just want to **debug** your script on single GPU. If so, simply after setting the `config.yaml` file
run the code by `python train_{task}.py`. It should be noted that accelerate supports single gpu and distributed training. So, you can use it for your 
final training.

### Extract sequence representation
There are two related scripts for generating protein sequence embeddings from a pre-trained S-PLM: 
- **`extract_sequence_representation.py`** — intended for **small-scale modifications or debugging**,  
  allowing you to quickly run embedding generation for a few proteins directly within the script.
**`cli_seq_embed.py`** — designed for **batch processing** of protein sequences in a FASTA file.  
  It reads multiple sequences and outputs the embeddings to a pickle file. It supports both **protein-level** and **residue-level** representations.
###  Usage & Key Arguments
To generate embeddings, run:
```bash
python cli_seq_embed.py   --input_seq ./test.fasta   --config_path ./configs/representation_config.yaml   --checkpoint_path /path/to/checkpoint.pth   --result_path ./out
```
This produces a pickle file such as `protein_embeddings.pkl` containing a dictionary that maps **protein IDs → NumPy embedding arrays**.
| Argument | Description |
|-----------|-------------|
| `--input_seq` | Path to the input FASTA file containing protein sequences. |
| `--config_path` | Path to the model configuration YAML file. |
| `--checkpoint_path` | Optional path to pretrained model checkpoint. |
| `--result_path` | Output directory for saving embeddings. |
| `--out_file` | Output file name, default `protein_embeddings.pkl`. |
| `--residue_level` | Output residue level embeddings per amino acid. |
| `--truncate_inference` | Enable sequence truncation with 1 or 0. |
| `--max_length_inference` | Maximum sequence length when truncation is enabled. |
| `--afterproject` | Output post projection embeddings if supported by the model. |
---
### Examples
**Protein level embeddings**
```bash
python cli_seq_embed.py --input_seq sample.fasta   -c configs/representation_config.yaml   --checkpoint_path checkpoints/model.pth   --result_path ./out
```
**Truncated inference**
```bash
python cli_seq_embed.py --input_seq sample.fasta   -c configs/representation_config.yaml   --checkpoint_path checkpoints/model.pth   --result_path ./out   --truncate_inference 1   --max_length_inference 1022
```
**Residue level embeddings**
```bash
python cli_seq_embed.py --input_seq sample.fasta   -c configs/representation_config.yaml   --checkpoint_path checkpoints/model.pth   --result_path ./out   --residue_level
```
---
### 💾 Output Format
The output pickle file contains a Python dictionary:
```python
{
  "protein_id_1": np.ndarray,  # shape [embedding_dim] or [seq_len, embedding_dim]
  "protein_id_2": np.ndarray,
  ...
}
```
Each value corresponds to either a protein level or residue level embedding depending on your arguments.
---

## S-PLM Pretraining
For advanced users who wish to pretrain S-PLM from scratch, please refer to the [pretrain](https://github.com/duolinwang/S_PLM1-pretrain/tree/main)

## Citation
If you use this code or the pretrained models, please cite the following paper:
### [1] S-PLM V1: protein-level contrastive learning, using Swin-transformer as protein structure encoder.
Wang D, Pourmirzaei M, Abbas UL, Zeng S, Manshour N, Esmaili F, Poudel B, Jiang Y, Shao Q, Chen J, Xu D. S-PLM: Structure-Aware Protein Language Model via Contrastive Learning Between Sequence and Structure. Adv Sci (Weinh). 2025 Feb;12(5):e2404212. doi: 10.1002/advs.202404212. Epub 2024 Dec 12. PMID: 39665266; PMCID: PMC11791933.
### [2] S-PLM V2: updated residue level model, using GVP as protein structure encoder.
```bibtex
@article {Zhang2025.04.23.650337,
	author = {Zhang, Yichuan and Qin, Yongfang and Pourmirzaei, Mahdi and Shao, Qing and Wang, Duolin and Xu, Dong},
	title = {Enhancing Structure-aware Protein Language Models with Efficient Fine-tuning for Various Protein Prediction Tasks},
	elocation-id = {2025.04.23.650337},
	year = {2025},
	doi = {10.1101/2025.04.23.650337},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Proteins are crucial in a wide range of biological and engineering processes. Large protein language models (PLMs) can significantly advance our understanding and engineering of proteins. However, the effectiveness of PLMs in prediction and design is largely based on the representations derived from protein sequences. Without incorporating the three-dimensional structures of proteins, PLMs would overlook crucial aspects of how proteins interact with other molecules, thereby limiting their predictive accuracy. To address this issue, we present S-PLM, a 3D structure-aware PLM that employs multi-view contrastive learning to align protein sequences with their 3D structures in a unified latent space. Previously, we utilized a contact map-based approach to encode structural information, applying the Swin-Transformer to contact maps derived from AlphaFold-predicted protein structures. This work introduces a new approach that leverages a Geometric Vector Perceptron (GVP) model to process 3D coordinates and obtain structural embeddings. We focus on the application of structure-aware models for protein-related tasks by utilizing efficient fine-tuning methods to achieve optimal performance without significant computational costs. Our results show that S-PLM outperforms sequence-only PLMs across all protein clustering and classification tasks, achieving performance on par with state-of-the-art methods that require both sequence and structure inputs. S-PLM and its tuning tools are available at https://github.com/duolinwang/S-PLM/.Competing Interest StatementThe authors have declared no competing interest.National Institutes of Health, , R35GM126985, R01LM014510National Science Foundation, , 2138259, 2138286, 2138307, 2137603, 2138296},
	URL = {https://www.biorxiv.org/content/early/2025/04/26/2025.04.23.650337},
	eprint = {https://www.biorxiv.org/content/early/2025/04/26/2025.04.23.650337.full.pdf},
	journal = {bioRxiv}
}
