#!/usr/bin/env python
import os
import torch
from torch.optim import Adam
from dataset import onehot
from torch.functional import F
import numpy as np
from model import CNNClassifier
import argparse
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger('predict promoter')


def main():
    parser = argparse.ArgumentParser(description='identify position of best 50 mer in input sequences')
    parser.add_argument('--fasta','-f',type=str, required=True, help="Input sequence")
    parser.add_argument('--batch-size','-bs',type=int,default=256,help="Batch size for scanning")
    parser.add_argument('--device','-d',default="cuda:1",choices=["cuda:0","cuda:1","cpu"],help="Device to run the model")
    parser.add_argument('--model','-m', default = "models/50.32.5.5.pt", type=str,help="Where to load the model parameters")
    parser.add_argument('--output','-o',required=True,type=str,help="Where to save the predicted probabilities")
    parser.add_argument('--n-channels', '-c', type=int, default=32, help="number of channels to use")
    parser.add_argument('--kernel-size', '-k', type=int, default=5, help="convolution kernel size in the res-block")
    parser.add_argument('--length', '-l', type=int, default=50, help="length of an instance")
    args = parser.parse_args()
    sequences = []
    seq_ids = []
    logger.info("load sequence ...")
    with open(args.fasta) as f:
        for header in f:
            seq_id = header[1:].strip().split(" ")[0]
            sequence = next(f).strip()
            seq_ids.append(seq_id)
            sequences.append(sequence)

    logger.info(f"load model weights from {args.model} ...")
    model = CNNClassifier(n_channels=args.n_channels, kernel_size = args.kernel_size)
    model = model.to(args.device)
    model = model.eval()
    state_dict = torch.load(args.model, map_location = args.device)
    model.load_state_dict(state_dict) 



    batched_instances = []
    batched_positions = []

    best_positions = {}
    best_scores = {}

    n = 0
    for seq_id, sequence in zip(seq_ids, sequences):
        if (n+1)%1000 == 0:
            logger.info(f"{round(n/1000,2)} K sequence processed")
        n += 1
        L = len(sequence)
        if L < args.length:
            #logger.info(f"{seq_id} is shorter than {args.length} nt, skip it .")
            continue
        sequence = onehot(sequence)
        p = 0
        while p + args.length <= L:
            instance = sequence[:,:,p:p+args.length]
            batched_instances.append(instance)
            batched_positions.append((seq_id, p))
            p += 1
            if len(batched_instances) == args.batch_size:
                X = torch.cat(batched_instances).to(args.device)
                scores = torch.exp(model(X)[:,1]).detach().cpu().numpy()
                for i, (seq_id, p) in enumerate(batched_positions):
                    score = scores[i]
                    if (seq_id not in best_scores) or (score > best_scores[seq_id]):
                        best_scores[seq_id] = score
                        best_positions[seq_id] = p
                batched_instances = []
                batched_positions = []

    if len(batched_instances) > 0:
        X = torch.cat(batched_instances).to(args.device)
        scores = torch.exp(model(X)[:,1]).detach().cpu().numpy()
        for i, (seq_id, p) in enumerate(batched_positions):
            score = scores[i]
            if (seq_id not in best_scores) or (score > best_scores[seq_id]):
                best_scores[seq_id] = score
                best_positions[seq_id] = p

    logger.info(f"saving results to {args.output} ...")
    fout = open(args.output,"w")

    for i,seq_id in enumerate(seq_ids):
        if seq_id not in best_positions:
            continue
        p = best_positions[seq_id]   
        sequence = sequences[i][p:p+args.length]
        print(seq_id, p, p+args.length, sequence, best_scores[seq_id],"+" , sep="\t",file=fout)

    fout.close()
    logger.info("all done .")

                
if __name__ == "__main__":
    main()
