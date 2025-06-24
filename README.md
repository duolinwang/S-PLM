# S-PLM: Structure-aware Protein Language Model via Contrastive Learning between Sequence and Structure

<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Oryx Video-ChatGPT">
</p>

[![paper](https://img.shields.io/badge/bioRxiv-Paper-<COLOR>.svg)](https://www.biorxiv.org/content/10.1101/2023.08.06.552203v3)

This is the official implementation of S-PLM paper (S-PLM V1). S-PLM is a 3D structure-aware protein language model (PLM) that enables the sequence-based embedding to carry the structural information through multi-view contrastive learning. 

This repository offers comprehensive guidance on utilizing pre-trained S-PLM models to generate structure-aware protein representations. Additionally, it provides a library of code for implementing lightweight tuning methods tailored for various downstream supervised learning tasks involving proteins.

The tasks include Enzyme Commission number (EC) prediction, Gene Ontology (GO) prediction, protein fold (fold) and enzyme reaction (ER) classification, and protein secondary structure (SS) prediction. 

The lightweight tunning methods include fine-tune top layers, Adapter Tuning, and Low-rank adaptation (LoRA). Users can train a task-specific model using different tuning methods by modifying the configuration files provided in the [configs directory](https://github.com/duolinwang/S-PLM/tree/main/configs)
## To use an updated residue-level pre-trained model: S-PLM V2, refer to [model explaination](https://github.com/duolinwang/S-PLM/blob/main/model/Model%20readme.md)
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
### Use S-PLM for downstream tasks
To utilize the accelerator power in you training code such as distributed multi GPU training, you have to set the accelerator config by running `accelearte config` in the command line.
Then, you have to set the training settings and hyperparameters inside your target task `configs/config_{task}.yaml` file.
Finally, you can start your training for downstream tasks using a config file from configs and a pretrained [S-PLM model](https://mailmissouri-my.sharepoint.com/:f:/g/personal/wangdu_umsystem_edu/Evk7BBT5LxRMpsHzKxmi0DEBrgv1mgBK0MRuRHJSqSoHZQ?e=Eozrwh) by running
```commandline
accelerate launch train_{task}.py --config_path configs/<config_name> --resume_path model/checkpoint_0520000.pth`
```

### Examples 
#### Train and evaluate of GO prediction tasks
```commandline
accelerate launch train_go.py --config_path configs/bp_config_adapterH_adapterH.yaml --resume_path model/checkpoint_0520000.pth
accelerate launch train_go.py --config_path configs/cc_config_adapterH_adapterH.yaml --resume_path model/checkpoint_0520000.pth
accelerate launch train_go.py --config_path configs/mf_config_adapterH_adapterH.yaml --resume_path model/checkpoint_0520000.pth
```
#### Train and evaluate of fold classification
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

#### Extract sequence representation
To extract the protein sequence representation from a pre-trained S-PLM, you can use the `extract_sequence_representation.py`
similar to the following code:

```python
    import yaml
    from utils import load_configs, load_checkpoints_only
    from model import SequenceRepresentation
    
    # Create a list of protein sequences
    sequences = ["MHHHHHHSSGVDLGTENLYFQSNAMDFPQQLEA", "CVKQANQALSRFIAPLPFQNTPVVE", "TMQYGALLGGKRLR"]

    # Load the configuration file
    config_path = "./configs/representation_config.yaml"
    with open(config_path) as file:
        dict_config = yaml.full_load(file)
    configs = load_configs(dict_config)

    # Create the model using the configuration file
    model = SequenceRepresentation(logging=None, configs=configs)
    model.eval()
    # Load the S-PLM checkpoint file
    checkpoint_path = "your checkpoint_path"
    load_checkpoints_only(checkpoint_path, model)

    esm2_seq = [(i, str(sequences[i])) for i in range(len(sequences))]
    batch_labels, batch_strs, batch_tokens = model.batch_converter(esm2_seq)
    
    # Get the protein representation and residue representation
    protein_representation, residue_representation,mask = model(batch_tokens)
```
## S-PLM Pretraining
For advanced users who wish to pretrain S-PLM from scratch, please refer to the [pretrain](https://github.com/duolinwang/S_PLM1-pretrain/tree/main)

## 📜 Citation
If you use this code or the pretrained models, please cite the following paper:
### 1 S-PLM V1: protein-level contrastive learning, using Swin-transformer as protein structure encoder.
Wang D, Pourmirzaei M, Abbas UL, Zeng S, Manshour N, Esmaili F, Poudel B, Jiang Y, Shao Q, Chen J, Xu D. S-PLM: Structure-Aware Protein Language Model via Contrastive Learning Between Sequence and Structure. Adv Sci (Weinh). 2025 Feb;12(5):e2404212. doi: 10.1002/advs.202404212. Epub 2024 Dec 12. PMID: 39665266; PMCID: PMC11791933.
### 2 S-PLM V2: updated residue level model, using GVP as protein structure encoder.
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
