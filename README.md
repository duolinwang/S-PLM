# S-PLM
S-PLM: Structure-aware Protein Language Model via Contrastive Learning between Sequence and Structure

## Installation
To use S-PLM, please install the required packages as follows:

### 1 Use python environment:
Using `requirements.txt`
1. Create a python environment: `python3 -m venv <env_name>`.
2. Activate the environment you have just created: `source <env_name>/bin/activate`.
3. install dependencies inside it: `pip3 install -r requirements.txt`.

### 2 Use conda environment:
Using `environment.yml`
1. Create a new environment using the `environment.yml` file: `conda <env_name> create -f environment.yml`.
2. Activate the environment you have just created: `conda activate <env_name>`.

### 3 Manually install each dependency.
```sh
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install torchmetrics
conda install matplotlib
pip install accelerate
pip install transformers
pip install timm
pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
pip install tensorboard
pip install python-box
conda install pandas
pip install peft
pip install -U scikit-learn
pip install proteinshake
```
## Run
### Using S-PLM for downstream tasks
To utilize the accelerator power in you training code such as distributed multi GPU training, you have to set the accelerator config by running `accelearte config` in the command line.
Then, you have to set the training settings and hyperparameters inside your target task `configs/config_{task}.yaml` file.
Finally, you can start your training for downstream tasks using a config file from configs and a pretrained [S-PLM model](https://mailmissouri-my.sharepoint.com/:f:/g/personal/wangdu_umsystem_edu/Evk7BBT5LxRMpsHzKxmi0DEBrgv1mgBK0MRuRHJSqSoHZQ?e=Eozrwh) by running
```sh
accelerate launch train_{task}.py --config_path configs/<config_name> --resume_path model/checkpoint_0520000.pth`
```

### Examples 
#### Training and evaluation for GO prediction tasks:
```sh
accelerate launch train_go.py --config_path configs/bp_config_adapterH_adapterH.yaml --resume_path model/checkpoint_0520000.pth
accelerate launch train_go.py --config_path configs/cc_config_adapterH_adapterH.yaml --resume_path model/checkpoint_0520000.pth
accelerate launch train_go.py --config_path configs/mf_config_adapterH_adapterH.yaml --resume_path model/checkpoint_0520000.pth
```
#### Training and evaluation for fold classification:
```sh
accelerate launch train_fold.py --config_path configs/fold_config_adapterH_finetune.yaml --resume_path model/checkpoint_0520000.pth
```
#### Training and evaluation for secondary structure prediction:
```sh
accelerate launch train_ss.py --config_path configs/ss_config_adapterH_finetune.yaml --resume_path model/checkpoint_0520000.pth
```


You might not use accelerator to run the `train.py` script if you just want to **debug** your script on single GPU. If so, simply after setting the `config.yaml` file
run the code by `python train_{task}.py`. It should be noted that accelerate supports single gpu and distributed training. So, you can use it for your 
final training.
