import yaml
from utils import load_configs, load_checkpoints_only
from model import SequenceRepresentation


def main():
    # Create a list of protein sequences
    sequences = ["MHHHHHHSSGVDLGTENLYFQSNAMDFPQQLEA", "CVKQANQALSRFIAPLPFQNTPVVE", "TMQYGALLGGKRLR"]

    # Load the configuration file
    config_path = "./configs/representation_config.yaml"
    with open(config_path) as file:
        dict_config = yaml.full_load(file)
    configs = load_configs(dict_config)

    # Create the model using the configuration file
    model = SequenceRepresentation(logging=None, configs=configs)

    # Load the S-PLM checkpoint file
    # If the checkpoint is not loaded correctly, it will be ESM2 with randomly initialized adapterH will be used.
    # The config file should contain adapterH with the consistent number of adapter layers.
    checkpoint_path = "your checkpoint_path"
    load_checkpoints_only(checkpoint_path, model)

    esm2_seq = [(range(len(sequences)), str(sequences[i])) for i in range(len(sequences))]
    batch_labels, batch_strs, batch_tokens = model.batch_converter(esm2_seq)

    # Get the protein representation and residue representation
    protein_representation, residue_representation,mask = model(batch_tokens)


if __name__ == '__main__':
    main()
