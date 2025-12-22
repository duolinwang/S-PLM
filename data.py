import torch
import yaml
import numpy as np
import pandas as pd
import csv
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.utils import load_configs, calculate_class_weights
from transformers import AutoTokenizer
from utils.utils import truncate_seq


def load_EC_seq_annot(file_seq, file_annot):
    # Load EC annotations """
    # Example:
    # file_seq: EC_valid.csv
    # file_annot: EC_annot.csv
    prot2seq = {}
    prot2annot = {}

    # gain the annotation
    with open(file_annot, mode='r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        # molecular function
        next(reader, None)  # skip the headers
        ec_numbers = {'ec': next(reader)}
        next(reader, None)  # skip the headers
        counts = {'ec': np.zeros(len(ec_numbers['ec']), dtype=float)}
        for row in reader:
            prot, prot_ec_numbers = row[0], row[1]
            ec_indices = [ec_numbers['ec'].index(ec_num) for ec_num in prot_ec_numbers.split(',')]
            prot2annot[prot] = {'ec': np.zeros(len(ec_numbers['ec']), dtype=float)}
            prot2annot[prot]['ec'][ec_indices] = 1.0
            counts['ec'][ec_indices] += 1

    with open(file_seq, mode="r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        # molecular function
        next(reader, None)
        for row in reader:
            prot, prot_seq = row[0], row[1]
            prot2seq[prot] = {'seq': prot_seq}

    return prot2seq, prot2annot, ec_numbers, counts


def load_GO_seq_annot(file_seq, file_annot):
    # Load GO annotations """
    # Example:
    # file_seq: GO_valid.csv
    # file_annot: nrPDB-GO_annot.tsv
    # mf: 489, bp: 1943, cc: 320
    prot2seq = {}
    with open(file_seq, mode="r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        # molecular function
        next(reader, None)
        for row in reader:
            prot, prot_seq = row[0], row[1]
            prot2seq[prot] = {'seq': prot_seq}

    onts = ['mf', 'bp', 'cc']
    prot2annot = {}
    goterms = {ont: [] for ont in onts}
    gonames = {ont: [] for ont in onts}

    with open(file_annot, mode='r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')

        # molecular function
        next(reader, None)  # skip the headers
        goterms[onts[0]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[0]] = next(reader)

        # biological process
        next(reader, None)  # skip the headers
        goterms[onts[1]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[1]] = next(reader)

        # cellular component
        next(reader, None)  # skip the headers
        goterms[onts[2]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[2]] = next(reader)

        next(reader, None)  # skip the headers
        counts = {ont: np.zeros(len(goterms[ont]), dtype=float) for ont in onts}
        for row in reader:
            prot, prot_goterms = row[0], row[1:]
            prot2annot[prot] = {ont: [] for ont in onts}
            for i in range(3):
                goterm_indices = [goterms[onts[i]].index(goterm) for goterm in prot_goterms[i].split(',') if
                                  goterm != '']
                prot2annot[prot][onts[i]] = np.zeros(len(goterms[onts[i]]))
                prot2annot[prot][onts[i]][goterm_indices] = 1.0
                counts[onts[i]][goterm_indices] += 1.0

    return prot2seq, prot2annot, goterms, gonames, counts


# creat the ECDataset based on FoldDataset - QShao Dec-7-2023
class GODataset(Dataset):
    def __init__(self, type, configs):  # type must be "train", "test" or "valid"
        self.max_length = configs.encoder.max_len
        if type == 'train':
            self.samples = self.prepare_samples(configs.train_settings.data_path, configs.train_settings.label_path,
                                                configs)
        elif type == 'test':
            self.samples = self.prepare_samples(configs.test_settings.data_path, configs.test_settings.label_path,
                                                configs)
        elif type == 'valid':
            self.samples = self.prepare_samples(configs.valid_settings.data_path, configs.valid_settings.label_path,
                                                configs)
            # Below two line should be revised to count label weights QShao Dec-8-2023
        # self.multilabel_weights = calculate_label_weights(self.count_samples_by_multilabel(self.samples))  # How should we calculate this?
        # self.multilabel_weights = self.independent_exponential_smoothing(self.multilabel_weights)

        self.encoder_tokenizer = AutoTokenizer.from_pretrained(configs.encoder.model_name)

    @staticmethod
    def prepare_samples(f_seq, f_annot, configs):  # attention QShao Dec-7-2023
        """
        Revised by QShao Dec-7-2023.
        f_seq: EC_test.csv
        f_annot: nrPDB-EC_annot.tsv
        read seq from f_seq and label from f_annot and merge them
        into a pandas dataframe df
        Returns:
        list: A list of tuples, where each tuple is (sequence, label).
        """
        prot2seq, prot2annot, goterms, gonames, counts = load_GO_seq_annot(f_seq, f_annot)

        df = pd.DataFrame(columns=['sequence', 'label'])
        for item in prot2seq:
            new_row = {'sequence': prot2seq[item]['seq'], 'label': prot2annot[item][configs.encoder.task]}
            df = df._append(new_row, ignore_index=True)

        return list(zip(df['sequence'], df['label']))

    @staticmethod
    def count_samples_by_multilabel(samples):  # This function should be revised to count label weights QShao Dec-8-2023
        """
        Count the number of samples for each label.
        If one sample has multiple labels. It should be
        """
        label_counts = {}
        # Iterate over the samples
        for _, multilabel in samples:
            label_nonzero_index = np.nonzero(multilabel)
            for i in range(label_nonzero_index):
                label = multilabel(label_nonzero_index[i])
                if label not in label_counts:
                    label_counts[label] = 0
                label_counts[label] += 1

        return label_counts

    @staticmethod
    def independent_exponential_smoothing(weights_dict, alpha=0.5):
        """
        Apply independent exponential smoothing to the weights.
        Each weight is reduced by an exponential decay factor.
        alpha is the smoothing factor where 0 < alpha <= 1.
        """
        # Apply independent exponential smoothing to each weight
        smoothed_weights = {k: v ** alpha for k, v in weights_dict.items()}

        return smoothed_weights

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sequence, label_idx = self.samples[
            idx]  # sample_weight = self.class_weights[label_idx] QShao Dec-7-2023 Obsolete because we do not use weight
        encoded_sequence = self.encoder_tokenizer(sequence, max_length=self.max_length, padding='max_length',
                                                  truncation=True,
                                                  return_tensors="pt"
                                                  )
        encoded_sequence['input_ids'] = torch.squeeze(encoded_sequence['input_ids'])
        encoded_sequence['attention_mask'] = torch.squeeze(encoded_sequence['attention_mask'])

        return encoded_sequence, label_idx


class FoldDataset(Dataset):
    def __init__(self, df, configs):
        self.samples = self.prepare_samples(df)
        self.max_length = configs.encoder.max_len
        self.class_weights = calculate_class_weights(self.count_samples_by_class(self.samples))
        self.class_weights = self.independent_exponential_smoothing(self.class_weights)

        self.encoder_tokenizer = AutoTokenizer.from_pretrained(configs.encoder.model_name)

    @staticmethod
    def prepare_samples(df):
        """
        Convert a pandas DataFrame with 'sequence' and 'label' columns into a list of tuples.

        Parameters:
        df (pandas.DataFrame): DataFrame with at least two columns 'sequence' and 'label'.

        Returns:
        list: A list of tuples, where each tuple is (sequence, label).
        """
        return list(zip(df['sequence'], df['label']))

    @staticmethod
    def count_samples_by_class(samples):
        """Count the number of samples for each class."""
        class_counts = {}

        # Iterate over the samples
        for _, label in samples:
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1

        return class_counts

    @staticmethod
    def independent_exponential_smoothing(weights_dict, alpha=0.5):
        """
        Apply independent exponential smoothing to the weights.
        Each weight is reduced by an exponential decay factor.
        alpha is the smoothing factor where 0 < alpha <= 1.
        """
        # Apply independent exponential smoothing to each weight
        smoothed_weights = {k: v ** alpha for k, v in weights_dict.items()}

        return smoothed_weights

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sequence, label_idx = self.samples[idx]
        sample_weight = self.class_weights[label_idx]

        encoded_sequence = self.encoder_tokenizer(sequence, max_length=self.max_length, padding='max_length',
                                                  truncation=True,
                                                  return_tensors="pt"
                                                  )
        encoded_sequence['input_ids'] = torch.squeeze(encoded_sequence['input_ids'])
        encoded_sequence['attention_mask'] = torch.squeeze(encoded_sequence['attention_mask'])

        return encoded_sequence, label_idx, sample_weight


class ERDataset(Dataset):
    def __init__(self, df, train_label_index_mapping, configs):
        self.samples = self.prepare_samples(df, train_label_index_mapping)
        self.max_length = configs.encoder.max_len
        self.class_weights = calculate_class_weights(self.count_samples_by_class(self.samples))
        self.class_weights = self.independent_exponential_smoothing(self.class_weights)

        self.encoder_tokenizer = AutoTokenizer.from_pretrained(configs.encoder.model_name)

    @staticmethod
    def prepare_samples(df, label_index_mapping):
        """
        Convert a pandas DataFrame with 'sequence' and 'label' columns into a list of tuples,
        with class strings converted to indexes using an optional pre-defined mapping.

        Parameters:
        df (pandas.DataFrame): DataFrame with at least two columns 'sequence' and 'label'.
        label_index_mapping (dict, optional): Mapping of labels to indexes. If None, a new mapping is created.

        Returns:
        list: A list of tuples (sequence, label index).
        """
        # if label_index_mapping is None:
        #     # Create new label-index mapping
        #     labels, unique_labels = pd.factorize(df['label'])
        #     label_index_mapping = {label: index for index, label in enumerate(unique_labels)}
        # else:
        # Use existing label-index mapping
        labels = df['label'].map(label_index_mapping).fillna(-1).astype(int)

        return list(zip(df['sequence'], labels))

    @staticmethod
    def count_samples_by_class(samples):
        """Count the number of samples for each class."""
        class_counts = {}

        # Iterate over the samples
        for _, label in samples:
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1

        return class_counts

    @staticmethod
    def independent_exponential_smoothing(weights_dict, alpha=0.5):
        """
        Apply independent exponential smoothing to the weights.
        Each weight is reduced by an exponential decay factor.
        alpha is the smoothing factor where 0 < alpha <= 1.
        """
        # Apply independent exponential smoothing to each weight
        smoothed_weights = {k: v ** alpha for k, v in weights_dict.items()}

        return smoothed_weights

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sequence, label_idx = self.samples[idx]
        sample_weight = self.class_weights[label_idx]

        encoded_sequence = self.encoder_tokenizer(sequence, max_length=self.max_length, padding='max_length',
                                                  truncation=True,
                                                  return_tensors="pt"
                                                  )
        encoded_sequence['input_ids'] = torch.squeeze(encoded_sequence['input_ids'])
        encoded_sequence['attention_mask'] = torch.squeeze(encoded_sequence['attention_mask'])

        return encoded_sequence, label_idx, sample_weight


class SSDataset(Dataset):
    def __init__(self, dataset_path, configs):
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(configs.encoder.model_name)
        self.max_length = configs.encoder.max_len
        self.samples = self.process_sequences(dataset_path)

    @staticmethod
    def process_sequences(file_path):
        """
        Processes a CSV file containing concatenated protein sequences and their labels.

        This function reads a CSV file where each row in the 'sequence label' column
        contains a protein sequence followed by its corresponding label. The function
        separates each sequence from its label, ensuring that the lengths of the sequence
        and the label are identical. It returns a list of tuples, each containing a sequence
        and its matching label.

        Parameters:
        file_path (str): The path to the CSV file.

        Returns:
        list of tuples: A list where each tuple contains a sequence and its corresponding label.
        """

        data = pd.read_csv(file_path)
        sequence_label_pairs = []

        for row in data["sequence label"]:
            # Separating alphabetic characters (sequence) from numeric characters (label)
            sequence = ''.join(filter(str.isalpha, row))
            label = ''.join(filter(str.isdigit, row))

            # Ensure the sequence and label lengths are identical
            if len(sequence) == len(label):
                sequence_label_pairs.append((sequence, label))
            else:
                # If lengths are not identical, print the sequence and label for inspection
                print(f"Sequence: {sequence}, Label: {label} (Lengths not equal)")

        return sequence_label_pairs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sequence, label = self.samples[index]

        label = label[:self.max_length]
        encoded_sequence = self.encoder_tokenizer(sequence, max_length=self.max_length + 2, padding='max_length',
                                                  truncation=True,
                                                  return_tensors="pt"
                                                  )
        encoded_sequence['input_ids'] = torch.squeeze(encoded_sequence['input_ids'])
        encoded_sequence['attention_mask'] = torch.squeeze(encoded_sequence['attention_mask'])

        length = len(label)
        label_list = [int(char) for char in label if char.isdigit()]
        label = np.array(label_list)
        padded_label = np.pad(label, (0, self.max_length - len(label)), 'constant')
        padded_label = torch.from_numpy(padded_label)
        mask = torch.zeros((self.max_length,), dtype=torch.bool)
        mask[:length] = 1
        return encoded_sequence, padded_label, mask


class PTMDataset(Dataset):
    def __init__(self, seqs, labels, valid_mask, configs):
        self.seqs = seqs
        self.labels = labels
        self.valid_mask = valid_mask
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(configs.encoder.model_name)
        self.max_length = configs.encoder.max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        sequence, label, mask, index = self.seqs[index], self.labels[index], self.valid_mask[index], index
        encoded_sequence = self.encoder_tokenizer(sequence, max_length=self.max_length + 2, padding='max_length',
                                                  truncation=True,
                                                  return_tensors="pt"
                                                  )
        encoded_sequence['input_ids'] = torch.squeeze(encoded_sequence['input_ids'])
        encoded_sequence['attention_mask'] = torch.squeeze(encoded_sequence['attention_mask'])

        padded_mask = np.pad(mask, (0, self.max_length - len(mask)), 'constant')
        padded_label = np.pad(label, (0, self.max_length - len(label)), 'constant')
        return encoded_sequence, padded_label, padded_mask.astype(bool), index


class ECDataset(Dataset):
    def __init__(self, type, configs):  # type must be "train", "test" or "valid"
        self.max_length = configs.encoder.max_len
        if type == 'train':
            self.samples = self.prepare_samples(configs.train_settings.data_path, configs.train_settings.label_path)
        elif type == 'test':
            self.samples = self.prepare_samples(configs.test_settings.data_path, configs.test_settings.label_path)
        elif type == 'valid':
            self.samples = self.prepare_samples(configs.valid_settings.data_path, configs.valid_settings.label_path)
            # Below two line should be revised to count label weights QShao Dec-8-2023
        # self.multilabel_weights = calculate_label_weights(self.count_samples_by_multilabel(self.samples))  # How should we calculate this?
        # self.multilabel_weights = self.independent_exponential_smoothing(self.multilabel_weights)

        self.encoder_tokenizer = AutoTokenizer.from_pretrained(configs.encoder.model_name)

    @staticmethod
    def prepare_samples(f_seq, f_annot):  # attention QShao Dec-7-2023
        """
        Revised by QShao Dec-7-2023.
        f_seq: EC_test.csv
        f_annot: nrPDB-EC_annot.tsv
        read seq from f_seq and label from f_annot and merge them
        into a pandas dataframe df
        Returns:
        list: A list of tuples, where each tuple is (sequence, label).
        """
        prot2seq, prot2annot, ec_numbers, counts = load_EC_seq_annot(f_seq, f_annot)

        df = pd.DataFrame(columns=['sequence', 'label'])
        for item in prot2seq:
            new_row = {'sequence': prot2seq[item]['seq'], 'label': prot2annot[item]['ec']}
            df = df._append(new_row, ignore_index=True)

        return list(zip(df['sequence'], df['label']))

    @staticmethod
    def count_samples_by_multilabel(samples):  # This function should be revised to count label weights QShao Dec-8-2023
        """
        Count the number of samples for each label.
        If one sample has multiple labels. It should be
        """
        label_counts = {}
        # Iterate over the samples
        for _, multilabel in samples:
            label_nonzero_index = np.nonzero(multilabel)
            for i in range(label_nonzero_index):
                label = multilabel(label_nonzero_index[i])
                if label not in label_counts:
                    label_counts[label] = 0
                label_counts[label] += 1

        return label_counts

    @staticmethod
    def independent_exponential_smoothing(weights_dict, alpha=0.5):
        """
        Apply independent exponential smoothing to the weights.
        Each weight is reduced by an exponential decay factor.
        alpha is the smoothing factor where 0 < alpha <= 1.
        """
        # Apply independent exponential smoothing to each weight
        smoothed_weights = {k: v ** alpha for k, v in weights_dict.items()}

        return smoothed_weights

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sequence, label_idx = self.samples[
            idx]  # sample_weight = self.class_weights[label_idx] QShao Dec-7-2023 Obsolete because we do not use weight
        encoded_sequence = self.encoder_tokenizer(sequence, max_length=self.max_length, padding='max_length',
                                                  truncation=True,
                                                  return_tensors="pt"
                                                  )
        encoded_sequence['input_ids'] = torch.squeeze(encoded_sequence['input_ids'])
        encoded_sequence['attention_mask'] = torch.squeeze(encoded_sequence['attention_mask'])

        return encoded_sequence, label_idx


def prepare_dataloaders_enzyme_commission(configs):
    train_dataset = ECDataset('train', configs=configs)  # QShao, Dec-7-2023
    valid_dataset = ECDataset('valid', configs=configs)
    test_dataset = ECDataset('test', configs=configs)

    train_dataloader = DataLoader(train_dataset, batch_size=configs.train_settings.batch_size,
                                  num_workers=configs.train_settings.num_workers,
                                  shuffle=configs.train_settings.shuffle)
    valid_dataloader = DataLoader(valid_dataset, batch_size=configs.valid_settings.batch_size,
                                  num_workers=configs.valid_settings.num_workers, shuffle=False)

    test_dataloader = DataLoader(test_dataset, batch_size=configs.valid_settings.batch_size,
                                 num_workers=configs.valid_settings.num_workers, shuffle=False)

    return {'train': train_dataloader, 'valid': valid_dataloader,
            'test': test_dataloader}


def prepare_dataloaders_ptm(configs):
    train = np.load(configs.train_settings.data_path, allow_pickle=True)
    val = np.load(configs.valid_settings.data_path, allow_pickle=True)
    test = np.load(configs.test_settings.data_path, allow_pickle=True)

    dataset_train = PTMDataset(truncate_seq(train['x'], configs.encoder.max_len),
                               truncate_seq(train['label'], configs.encoder.max_len),
                               truncate_seq(train['valid_mask'], configs.encoder.max_len), configs)

    dataset_val = PTMDataset(truncate_seq(val['x'], configs.encoder.max_len),
                             truncate_seq(val['label'], configs.encoder.max_len),
                             truncate_seq(val['valid_mask'], configs.encoder.max_len), configs)

    dataset_test = PTMDataset(truncate_seq(test['x'], configs.encoder.max_len),
                              truncate_seq(test['label'], configs.encoder.max_len),
                              truncate_seq(test['valid_mask'], configs.encoder.max_len), configs)

    train_loader = DataLoader(dataset_train, batch_size=configs.train_settings.batch_size,
                              shuffle=True, pin_memory=False, drop_last=False,
                              num_workers=configs.train_settings.num_workers)
    val_loader = DataLoader(dataset_val, batch_size=configs.valid_settings.batch_size,
                            shuffle=False, pin_memory=False, drop_last=False,
                            num_workers=configs.valid_settings.num_workers)
    test_loader = DataLoader(dataset_test, batch_size=configs.test_settings.batch_size,
                             shuffle=False, pin_memory=False, drop_last=False,
                             num_workers=configs.test_settings.num_workers)

    return {'train': train_loader, 'valid': val_loader, 'test': test_loader}


def prepare_dataloaders_secondary_structure(configs):
    dataset_train = SSDataset(configs.train_settings.data_path, configs)

    dataset_val = SSDataset(configs.valid_settings.data_path, configs)

    dataset_test = SSDataset(configs.test_settings.data_path, configs)

    train_loader = DataLoader(dataset_train, batch_size=configs.train_settings.batch_size,
                              shuffle=True, pin_memory=False, drop_last=False,
                              num_workers=configs.train_settings.num_workers)
    val_loader = DataLoader(dataset_val, batch_size=configs.valid_settings.batch_size,
                            shuffle=False, pin_memory=False, drop_last=False,
                            num_workers=configs.valid_settings.num_workers)
    test_loader = DataLoader(dataset_test, batch_size=configs.test_settings.batch_size,
                             shuffle=False, pin_memory=False, drop_last=False,
                             num_workers=configs.test_settings.num_workers)

    return {'train': train_loader, 'valid': val_loader, 'test': test_loader}


def prepare_dataloaders_go(configs):
    train_dataset = GODataset('train', configs=configs)  # QShao, Dec-7-2023
    valid_dataset = GODataset('valid', configs=configs)
    test_dataset = GODataset('test', configs=configs)

    train_dataloader = DataLoader(train_dataset, batch_size=configs.train_settings.batch_size,
                                  num_workers=configs.train_settings.num_workers,
                                  shuffle=configs.train_settings.shuffle)
    valid_dataloader = DataLoader(valid_dataset, batch_size=configs.valid_settings.batch_size,
                                  num_workers=configs.valid_settings.num_workers, shuffle=False)

    test_dataloader = DataLoader(test_dataset, batch_size=configs.valid_settings.batch_size,
                                 num_workers=configs.valid_settings.num_workers, shuffle=False)

    return {'train': train_dataloader, 'valid': valid_dataloader,
            'test': test_dataloader}


def prepare_dataloaders_enzyme_reaction(configs):
    train_df = pd.read_csv(configs.train_settings.data_path)
    valid_df = pd.read_csv(configs.valid_settings.data_path)
    test_df = pd.read_csv(configs.test_settings.data_path)

    train_label_index_mapping = {label: index for index, label in enumerate(train_df['label'].unique())}

    train_dataset = ERDataset(train_df, train_label_index_mapping, configs=configs)
    valid_dataset = ERDataset(valid_df, train_label_index_mapping, configs=configs)
    test_dataset_family = ERDataset(test_df, train_label_index_mapping, configs=configs)

    train_dataloader = DataLoader(train_dataset, batch_size=configs.train_settings.batch_size,
                                  num_workers=configs.train_settings.num_workers,
                                  shuffle=configs.train_settings.shuffle)
    valid_dataloader = DataLoader(valid_dataset, batch_size=configs.valid_settings.batch_size,
                                  num_workers=configs.valid_settings.num_workers, shuffle=False)

    test_dataloader = DataLoader(test_dataset_family, batch_size=configs.valid_settings.batch_size,
                                 num_workers=configs.valid_settings.num_workers, shuffle=False)

    return {'train': train_dataloader, 'valid': valid_dataloader,
            'test': test_dataloader}


def prepare_dataloaders_fold(configs):
    train_df = pd.read_csv(configs.train_settings.data_path)
    valid_df = pd.read_csv(configs.valid_settings.data_path)
    test_df_family = pd.read_csv(configs.test_settings.data_path_family)
    test_df_super_family = pd.read_csv(configs.test_settings.data_path_superfamily)
    test_df_fold = pd.read_csv(configs.test_settings.data_path_fold)

    train_dataset = FoldDataset(train_df, configs=configs)
    valid_dataset = FoldDataset(valid_df, configs=configs)
    test_dataset_family = FoldDataset(test_df_family, configs=configs)
    test_dataset_super_family = FoldDataset(test_df_super_family, configs=configs)
    test_dataset_fold = FoldDataset(test_df_fold, configs=configs)

    train_dataloader = DataLoader(train_dataset, batch_size=configs.train_settings.batch_size,
                                  num_workers=configs.train_settings.num_workers,
                                  shuffle=configs.train_settings.shuffle)
    valid_dataloader = DataLoader(valid_dataset, batch_size=configs.valid_settings.batch_size,
                                  num_workers=configs.valid_settings.num_workers, shuffle=False)

    test_dataloader_family = DataLoader(test_dataset_family, batch_size=configs.valid_settings.batch_size,
                                        num_workers=configs.valid_settings.num_workers, shuffle=False)
    test_dataloader_super_family = DataLoader(test_dataset_super_family, batch_size=configs.valid_settings.batch_size,
                                              num_workers=configs.valid_settings.num_workers, shuffle=False)
    test_dataloader_fold = DataLoader(test_dataset_fold, batch_size=configs.valid_settings.batch_size,
                                      num_workers=configs.valid_settings.num_workers, shuffle=False)

    return {'train': train_dataloader, 'valid': valid_dataloader,
            'test_family': test_dataloader_family, 'test_super_family': test_dataloader_super_family,
            'test_fold': test_dataloader_fold}


if __name__ == '__main__':
    config_path = './config_enzyme_reaction.yaml'
    with open(config_path) as file:
        configs_dict = yaml.full_load(file)

    configs_file = load_configs(configs_dict)

    dataloaders_dict = prepare_dataloaders_enzyme_reaction(configs_file)
    max_position_value = []
    amino_acid = []
    for batch in dataloaders_dict['train']:
        sequence_batch, label_batch, position_batch, weights_batch = batch
        # print(sequence_batch['input_ids'].shape)
        # print(label_batch.shape)
        # print(position_batch.shape)
        max_position_value.append(position_batch.squeeze().numpy().item())
        amino_acid.append(sequence_batch["input_ids"][0][position_batch.squeeze().numpy().item()].item())
    print(set(max_position_value))
    print([dataloaders_dict['train'].dataset.encoder_tokenizer.id_to_token(i) for i in set(amino_acid)])
    print('done')
