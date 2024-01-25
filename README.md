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

### 3 Follow the `install.txt` file to manually install each dependency.

## Run
### Using S-PLM for downstream tasks
To start a new training you have to set the accelerator config by running `accelearte config` in the command line after 
`cd` to the **merged_tasks** folder.
By doing that, you can utilize the accelerator power in you training code such as distributed multi GPU training.

Then, you have to set the training settings and hyperparameters inside your target task `config_{task}.yaml` file.
Finally, you can start your training by running
`accelerate launch train_{task}.py`.

You might not use accelerator to run the `train.py` script if you just want to **debug** your script on single GPU. If so, simply after setting the `config.yaml` file
run the code by `python train_{task}.py`.

It should be noted that accelerate supports single gpu and distributed training. So, you can use it for your 
final training.
