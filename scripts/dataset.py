from collections import defaultdict
import numpy as np
import os
import torch
from torch.utils.data import Dataset
import logging
import random
from tqdm import tqdm
import sys
import re
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)s] %(message)s')
logger = logging.getLogger("Load training instances")


def collate_fn(examples):
    sequences = []
    labels = []
    group_ids = []
    for sequence, label, group_id in examples:
        #print(sequence.shape)
        sequences.append(sequence)
        labels.append(label)
        group_ids.append(group_id)
    labels = torch.tensor(labels)
    # batch , channel , length
    sequences = torch.cat(sequences)
    return sequences, labels, group_ids

DNA_alp = dict(zip("ACGT",list(range(4))))
protein_alp = dict(zip("ACDEFGHIKLMNPQRSTVWY",list(range(20))))

def onehot(sequence,alp="DNA"):
    assert alp in ["DNA","protein"], "the alphabet should be either DNA or protein"
    if alp == "DNA":
        c2i = DNA_alp
        x = torch.zeros((4,len(sequence)))
    else:
        c2i = protein_alp 
        x = torch.zeros((20,len(sequence)))
    tokens = []
    indices = []
    for i,c in enumerate(sequence):
        if c in c2i:
            tokens.append(c2i[c])  
            indices.append(i)
    x[tokens,indices] = 1
    return x[None,:,:]


class PromoterSet(Dataset):
    """
    load data for terminator prediction
    input:
       promoter sequence
       background sequence
       grouping og genomic sequences
    """

    def __init__(self,promoters,background, positive_fraction=0.5, 
                 shuffled_fraction = 0, length=50, stratified = True):
        from ushuffle import shuffle
        self.positive_fraction = positive_fraction
        self.shuffled_fraction = shuffled_fraction
        logger.info("Load terminator sequences ...")
        self.promoters = defaultdict(list)
        with open(promoters) as f:
            for header in f:
                sequence = next(f).strip().upper().replace("U","T")
                if len(sequence) != length:
                    continue
                if stratified:
                    group_id = header[1:].split(":")[0]
                else:
                    group_id = "dummy"
                self.promoters[group_id].append(sequence)
        self.background = defaultdict(list)
        logger.info("Load background sequences ...")
        with open(background) as f:
           for header in f:
                sequence = next(f).strip().upper().replace("U","T")
                if len(sequence) != length:
                    continue
                if stratified:
                    group_id = header[1:].split(":")[0]
                else:
                    group_id = "dummy"
                self.background[group_id].append(sequence)
        self.group_ids = [ group_id for group_id in self.promoters if group_id in self.background]
        logger.info(f"{len(self.group_ids)} present in both promoter set and background set .")

    def __len__(self):
        return 100000

    def __getitem__(self,idx):
        group_id = self.group_ids[idx%len(self.group_ids)] #np.random.choice(self.group_ids)
        if random.random() < self.positive_fraction:
            sequence = self.promoters[group_id][np.random.randint(0,len(self.promoters[group_id]))]
            label = 1
        else:
            sequence = self.background[group_id][np.random.randint(0,len(self.background[group_id]))]
            if random.random() < self.shuffled_fraction:
                if random.random() < self.positive_fraction:
                    sequence = self.promoters[group_id][np.random.randint(0,len(self.promoters[group_id]))]
                sequence = shuffle(sequence.encode(),2).decode()
            label = 0
        sequence = onehot(sequence)
        return sequence, label, group_id

