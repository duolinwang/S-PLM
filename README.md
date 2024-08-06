# S-PLM: Structure-aware Protein Language Model via Contrastive Learning between Sequence and Structure

<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Oryx Video-ChatGPT">
</p>

[![paper](https://img.shields.io/badge/bioRxiv-Paper-<COLOR>.svg)](https://www.biorxiv.org/content/10.1101/2023.08.06.552203v3)

This is the official implementation of S-PLM paper. S-PLM is a 3D structure-aware protein language model (PLM) that enables the sequence-based embedding to carry the structural information through multi-view contrastive learning. 

This repository offers comprehensive guidance on utilizing pre-trained S-PLM models to generate structure-aware protein representations. Additionally, it provides a library of code for implementing lightweight tuning methods tailored for various downstream supervised learning tasks involving proteins.

The tasks include Enzyme Commission number (EC) prediction, Gene Ontology (GO) prediction, protein fold (fold) and enzyme reaction (ER) classification, and protein secondary structure (SS) prediction. 

The lightweight tunning methods include fine-tune top layers, Adapter Tuning, and Low-rank adaptation (LoRA). Users can train a task-specific model using different tuning methods by modifying the configuration files provided in the [configs directory](https://github.com/duolinwang/S-PLM/tree/main/configs)



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

    esm2_seq = [(range(len(sequences)), str(sequences[i])) for i in range(len(sequences))]
    batch_labels, batch_strs, batch_tokens = model.batch_converter(esm2_seq)
    
    # Get the protein representation and residue representation
    protein_representation, residue_representation = model(batch_tokens)
```

## 📜 Citation
If you use this code or the pretrained models, please cite the following paper:
```bibtex
@article {Wang2023.08.06.552203,
	author = {Duolin Wang and Mahdi Pourmirzaei and Usman L Abbas and Shuai Zeng and Negin Manshour and Farzaneh Esmaili and Biplab Poudel and Yuexu Jiang and Qing Shao and Jin Chen and Dong Xu},
	title = {S-PLM: Structure-aware Protein Language Model via Contrastive Learning between Sequence and Structure},
	elocation-id = {2023.08.06.552203},
	year = {2024},
	doi = {10.1101/2023.08.06.552203},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Large protein language models (PLMs) present excellent potential to reshape protein research by encoding the amino acid sequences into mathematical and biological meaningful embeddings. However, the lack of crucial 3D structure information in most PLMs restricts the prediction capacity of PLMs in various applications, especially those heavily depending on 3D structures. To address this issue, we introduce S-PLM, a 3D structure-aware PLM utilizing multi-view contrastive learning to align the sequence and 3D structure of a protein in a coordinate space. S-PLM applies Swin-Transformer on AlphaFold-predicted protein structures to embed the structural information and fuses it into sequence-based embedding from ESM2. Additionally, we provide a library of lightweight tuning tools to adapt S-PLM for diverse protein property prediction tasks. Our results demonstrate S-PLM{\textquoteright}s superior performance over sequence-only PLMs, achieving competitiveness in protein function prediction compared to state-of-the-art methods employing both sequence and structure inputs.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2024/01/28/2023.08.06.552203},
	eprint = {https://www.biorxiv.org/content/early/2024/01/28/2023.08.06.552203.full.pdf},
	journal = {bioRxiv}
}
```
