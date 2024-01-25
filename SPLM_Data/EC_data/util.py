import csv
import numpy as np
def load_EC_annot(filename):
    # Load EC annotations """
    prot2annot = {}
    with open(filename, mode='r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        
        # molecular function
        next(reader, None)  # skip the headers
        ec_numbers = {'ec': next(reader)}
        next(reader, None)  # skip the headers
        counts = {'ec': np.zeros(len(ec_numbers['ec']), dtype=float)}
        for row in reader:
            prot, prot_ec_numbers = row[0], row[1]
            ec_indices = [ec_numbers['ec'].index(ec_num) for ec_num in prot_ec_numbers.split(',')]
            prot2annot[prot] = {'ec': np.zeros(len(ec_numbers['ec']), dtype=np.int64)}
            prot2annot[prot]['ec'][ec_indices] = 1.0
            counts['ec'][ec_indices] += 1
    return prot2annot, ec_numbers, ec_numbers, counts


prot2annot, ec_numbers, ec_numbers, counts=load_EC_annot("nrPDB-EC_2020.04_annot.tsv")