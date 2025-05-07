import sys

import numpy as np
import torch
from tqdm import tqdm
SEQ_LEN = 512
# SEQ2IND = {"I": 0,
#            "L": 1,
#            "V": 2,
#            "F": 3,
#            "M": 4,
#            "C": 5,
#            "A": 6,
#            "G": 7,
#            "P": 8,
#            "T": 9,
#            "S": 10,
#            "Y": 11,
#            "W": 12,
#            "Q": 13,
#            "N": 14,
#            "H": 15,
#            "E": 16,
#            "D": 17,
#            "K": 18,
#            "R": 19,
#            "X": 20,
#            "O": 21,
#            "Z": 22,
#            "B": 23,
#            "U": 24,
#            "-": 25
#            }  # J = padding, * = any amino
# * change to Z
# - change to B
SEQ2IND = {"I": 0,
           "L": 1,
           "V": 2,
           "F": 3,
           "M": 4,
           "C": 5,
           "A": 6,
           "G": 7,
           "P": 8,
           "T": 9,
           "S": 10,
           "Y": 11,
           "W": 12,
           "Q": 13,
           "N": 14,
           "H": 15,
           "E": 16,
           "D": 17,
           "K": 18,
           "R": 19,
           "X": 20,
           "J": 21,
           "*": 22,
           "-": 23
           }  # J = padding, * = any amino

IND2SEQ = {ind: AA for AA, ind in SEQ2IND.items()}


def inds_to_seq(seq):
    return [IND2SEQ[i] for i in seq]


def seq_to_inds(seq):
    return [SEQ2IND[i] for i in seq]


def load_raw_giff_data(input_data):
    targs = list(torch.from_numpy(np.array(input_data["enrichment"])))
    targets = torch.cat([x.unsqueeze(dim=0).type("torch.FloatTensor") for x in targs], 0)
    label = list(torch.from_numpy(np.array(input_data["label"])))
    labels = torch.cat([x.unsqueeze(dim=0).type("torch.FloatTensor") for x in label], 0)

    reps = []
    seqs = list(input_data['CDR3'])
    for seq in seqs:
        reps.append(torch.tensor(seq_to_inds(seq)))
    reps = torch.cat([x.type("torch.LongTensor").unsqueeze(dim=0) for x in reps])

    return reps, targets, labels


def load_raw_pab1_data(input_data):
    targs = list(torch.from_numpy(np.array(input_data["fitness"])))
    targets = torch.cat([x.unsqueeze(dim=0).type("torch.FloatTensor") for x in targs], 0)

    label = list(torch.from_numpy(np.array(input_data["label"])))
    labels = torch.cat([x.unsqueeze(dim=0).type("torch.FloatTensor") for x in label], 0)

    reps = []
    seqs = list(input_data["seq"])
    for seq in seqs:
        seq += "J" * (77 - len(seq))
        reps.append(torch.tensor(seq_to_inds(seq)))
    reps = torch.cat([x.type("torch.LongTensor").unsqueeze(dim=0) for x in reps])
    return reps, targets, labels


def load_raw_bgl3_data(input_data):
    targs = list(torch.from_numpy(np.array(input_data["fitness"])))
    targets = torch.cat([x.unsqueeze(dim=0).type("torch.FloatTensor") for x in targs], 0)

    label = list(torch.from_numpy(np.array(input_data["label"])))
    labels = torch.cat([x.unsqueeze(dim=0).type("torch.FloatTensor") for x in label], 0)

    reps = []
    seqs = list(input_data["seq"])
    for seq in seqs:
        reps.append(torch.tensor(seq_to_inds(seq)))
    reps = torch.cat([x.type("torch.LongTensor").unsqueeze(dim=0) for x in reps])
    return reps, targets, labels


def load_raw_ube4b_data(input_data):
    targs = list(torch.from_numpy(np.array(input_data["fitness"])))
    targets = torch.cat([x.unsqueeze(dim=0).type("torch.FloatTensor") for x in targs], 0)

    label = list(torch.from_numpy(np.array(input_data["label"])))
    labels = torch.cat([x.unsqueeze(dim=0).type("torch.FloatTensor") for x in label], 0)

    reps = []
    seqs = list(input_data["seq"])
    for seq in seqs:
        reps.append(torch.tensor(seq_to_inds(seq)))
    reps = torch.cat([x.type("torch.LongTensor").unsqueeze(dim=0) for x in reps])
    return reps, targets, labels


def load_raw_nesp_data(input_data):
    targs = input_data["tm"].to_numpy()
    targs = list(torch.from_numpy(targs))
    targets = torch.cat([x.unsqueeze(dim=0).type("torch.FloatTensor") for x in targs], 0)

    labels = input_data["label"].to_numpy()
    labels = list(torch.from_numpy(labels))
    seq_label = torch.cat([x.unsqueeze(dim=0).type("torch.FloatTensor") for x in labels], 0)

    reps = []
    seqs = list(input_data["protein_sequence"])
    for seq in seqs:
        seq += "J" * (1000 - len(seq))
        reps.append(torch.tensor(seq_to_inds(seq)))
    reps = torch.cat([x.type("torch.LongTensor").unsqueeze(dim=0) for x in reps])

    return reps, targets, seq_label


def load_raw_HIS7_data(input_data):
    targs = input_data["fitness"].to_numpy()
    targs = list(torch.from_numpy(targs))
    targets = torch.cat([x.unsqueeze(dim=0).type("torch.FloatTensor") for x in targs], 0)

    labels = input_data["label"].to_numpy()
    labels = list(torch.from_numpy(labels))
    seq_label = torch.cat([x.unsqueeze(dim=0).type("torch.FloatTensor") for x in labels], 0)

    reps = []
    seqs = list(input_data["seq"])
    for seq in seqs:
        seq += "J" * (235 - len(seq))
        reps.append(torch.tensor(seq_to_inds(seq)))
    reps = torch.cat([x.type("torch.LongTensor").unsqueeze(dim=0) for x in reps])

    return reps, targets, seq_label


def load_raw_CAPSD_data(input_data):
    targs = input_data["fitness"].to_numpy()
    targs = list(torch.from_numpy(targs))
    targets = torch.cat([x.unsqueeze(dim=0).type("torch.FloatTensor") for x in targs], 0)

    labels = input_data["label"].to_numpy()
    labels = list(torch.from_numpy(labels))
    seq_label = torch.cat([x.unsqueeze(dim=0).type("torch.FloatTensor") for x in labels], 0)

    reps = []
    seqs = list(input_data["seq"])
    for seq in seqs:
        seq += "J" * (749 - len(seq))
        reps.append(torch.tensor(seq_to_inds(seq)))
    reps = torch.cat([x.type("torch.LongTensor").unsqueeze(dim=0) for x in reps])

    return reps, targets, seq_label


def load_raw_B1LPA6_data(input_data):
    targs = input_data["fitness"].to_numpy()
    targs = list(torch.from_numpy(targs))
    targets = torch.cat([x.unsqueeze(dim=0).type("torch.FloatTensor") for x in targs], 0)

    labels = input_data["label"].to_numpy()
    labels = list(torch.from_numpy(labels))
    seq_label = torch.cat([x.unsqueeze(dim=0).type("torch.FloatTensor") for x in labels], 0)

    reps = []
    seqs = list(input_data["seq"])
    for seq in seqs:
        seq += "J" * (95 - len(seq))
        reps.append(torch.tensor(seq_to_inds(seq)))
    reps = torch.cat([x.type("torch.LongTensor").unsqueeze(dim=0) for x in reps])

    return reps, targets, seq_label


def load_raw_GFP_data(input_data):
    targs = input_data["fitness"].to_numpy()
    targs = list(torch.from_numpy(targs))
    targets = torch.cat([x.unsqueeze(dim=0).type("torch.FloatTensor") for x in targs], 0)

    labels = input_data["label"].to_numpy()
    labels = list(torch.from_numpy(labels))
    seq_label = torch.cat([x.unsqueeze(dim=0).type("torch.FloatTensor") for x in labels], 0)

    reps = []
    seqs = list(input_data["seq"])
    for seq in seqs:
        seq += "J" * (95 - len(seq))
        reps.append(torch.tensor(seq_to_inds(seq)))
    reps = torch.cat([x.type("torch.LongTensor").unsqueeze(dim=0) for x in reps])

    return reps, targets, seq_label


def load_raw_TAPE_data(input_data):
    targs = input_data["log_fluorescence"].to_numpy()
    targs = list(torch.from_numpy(targs))
    targets = torch.cat([x.unsqueeze(dim=0).type("torch.FloatTensor") for x in targs], 0)

    labels = input_data["label"].to_numpy()
    labels = list(torch.from_numpy(labels))
    seq_label = torch.cat([x.unsqueeze(dim=0).type("torch.FloatTensor") for x in labels], 0)
    reps = []
    seqs = list(input_data["primary"])
    for seq in seqs:
        seq += "J" * (95 - len(seq))
        reps.append(torch.tensor(seq_to_inds(seq)))
    reps = torch.cat([x.type("torch.LongTensor").unsqueeze(dim=0) for x in reps])

    return reps, targets, seq_label


def load_MSA_data(input_data):
    targets = torch.zeros(len(input_data))
    seq_label = torch.zeros(len(input_data))

    reps = []
    seqs = list(input_data["seq"])
    for seq in seqs:
        reps.append(torch.tensor(seq_to_inds(seq)))
    reps = torch.cat([x.type("torch.LongTensor").unsqueeze(dim=0) for x in reps])

    return reps, targets, seq_label


def load_raw_MSA_data(input_data):
    targets = torch.zeros(len(input_data))
    seq_label = torch.zeros(len(input_data))

    reps = []
    seqs = list(input_data["seq"])
    for seq in seqs:
        seq += "J" * (500 - len(seq))
        reps.append(torch.tensor(seq_to_inds(seq)))
    reps = torch.cat([x.type("torch.LongTensor").unsqueeze(dim=0) for x in reps])

    return reps, targets, seq_label


def load_raw_swissprot_data(input_data):
    reps = []
    seqs = list(input_data["sequence"])
    for seq in tqdm(seqs):
        if len(seq) <= SEQ_LEN:
            seq += "-" * (SEQ_LEN - len(seq))
            reps.append(torch.tensor(seq_to_inds(seq)))
        else:
            start = np.random.randint(0, len(seq) - SEQ_LEN)
            seq = seq[start:start + SEQ_LEN]
            reps.append(torch.tensor(seq_to_inds(seq)))

    reps = torch.cat([x.type("torch.LongTensor").unsqueeze(dim=0) for x in reps])
    targets = torch.zeros(len(reps))
    seq_label = torch.zeros(len(reps))

    return reps, targets, seq_label


def load_raw_MDH_data(input_data):
    targets = torch.zeros(len(input_data))
    seq_label = torch.zeros(len(input_data))

    reps = []
    seqs = list(input_data["seq"])
    for seq in seqs:
        seq += "J" * (505 - len(seq))
        reps.append(torch.tensor(seq_to_inds(seq)))
    reps = torch.cat([x.type("torch.LongTensor").unsqueeze(dim=0) for x in reps])

    return reps, targets, seq_label
