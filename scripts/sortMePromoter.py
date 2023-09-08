#!/usr/bin/env python
import os
import torch
from dataset import onehot
from torch.functional import F
import numpy as np
from model import CNNClassifier
import argparse
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger('predict promoter')
from scipy.signal import convolve

rc_lut = {"A":"T","C":"G","G":"C","T":"A"}


def get_rc(sequence):
    rcs = []
    for c in sequence:
        if c in rc_lut:
            c = rc_lut[c]
        rcs.append(c)
    return "".join(rcs[::-1])

def main():
    parser = argparse.ArgumentParser(description='identify position of best 50 mer in input sequences')
    parser.add_argument('--fasta','-f',type=str, required=True, help="Input sequence")
    parser.add_argument('--batch-size','-bs',type=int,default=16384,help="Batch size for scanning")
    parser.add_argument('--device','-d',default="cuda:1",choices=["cuda:0","cuda:1","cpu"],help="Device to run the model")
    parser.add_argument('--model','-m', default = "models/37.pt", type=str,help="Where to load the model parameters")
    parser.add_argument('--output','-o',required=True,type=str,help="Where to save the predicted probabilities")
    parser.add_argument('--n-channels', '-nc', type=int, default=64, help="number of channels to use")
    parser.add_argument('--n-blocks', '-nb', type=int, default=10, help="number of blocks in the model")
    parser.add_argument('--kernel-size', '-k', type=int, default=5, help="convolution kernel size in the res-block")
    parser.add_argument('--length', '-l', type=int, default=50, help="length of an instance")
    parser.add_argument('--reverse-complementary', '-rc', action = "store_true", help="scoring both strand")
    parser.add_argument('--stride', '-s', type=int, default = 1, help="stride size for scanning")
    parser.add_argument('--cutoff', '-c', type=float, default = 0.0, help="cutoff")
    parser.add_argument('--smooth', type=int, default = 7, help="smoothing with in a sliding window. the default value 1 means no smoothing")
    parser.add_argument('--offset', type=int, default = 0, help="offset relative to predicted ends")
    args = parser.parse_args()
    sequences = []
    seq_ids = []
    logger.info("load sequence ...")
    with open(args.fasta) as f:
        for line in f:
            if line.startswith(">"):
                seq_id = line[1:].strip().split(" ")[0]
                seq_ids.append(seq_id)
                sequences.append("")
            else:
                sequence = line.strip().replace("U","T")
                sequences[-1] += sequence

    logger.info(f"load model weights from {args.model} ...")
    model = CNNClassifier(n_channels=args.n_channels, kernel_size = args.kernel_size, n_blocks=args.n_blocks)
    model = model.to(args.device)
    model = model.eval()
    state_dict = torch.load(args.model, map_location = args.device)
    model.load_state_dict(state_dict) 


    batched_instances = []
    batched_positions = []

    positions_by_sequence = {}
    scores_by_sequence = {}

    logger.info("run prediction ...")
    n = 0
    for seq_id, sequence in zip(seq_ids, sequences):
        if (n+1)%1000 == 0:
            logger.info(f"{round(n/1000)} K sequence processed")
        L = len(sequence)
        n += 1
        #L = len(sequence)
        if L < args.length:
            #logger.info(f"{seq_id} is shorter than {args.length} nt, skip it .")
            continue
        if args.reverse_complementary:
            rc_sequence = get_rc(sequence)
            rc_sequence = onehot(rc_sequence)
        sequence = onehot(sequence)
        p = 0
        while p + args.length <= L:
            instance = sequence[:,:,p:p+args.length]
            batched_instances.append(instance)
            batched_positions.append((seq_id, p,"+"))
            if args.reverse_complementary:
                instance = rc_sequence[:,:,L-p-args.length:L-p]
                batched_instances.append(instance)
                batched_positions.append((seq_id, p,"-"))
            p += args.stride
            if len(batched_instances) == args.batch_size:
                X = torch.cat(batched_instances).to(args.device)
                scores = torch.exp(model(X)[:,1]).detach().cpu().numpy()
                for i, (seq_id, position, strand) in enumerate(batched_positions):
                    score = scores[i]
                    #if score < args.cutoff:
                    #    continue
                    if (seq_id, strand) not in scores_by_sequence:
                        scores_by_sequence[(seq_id, strand)] = []
                        positions_by_sequence[(seq_id, strand)] = []
                    scores_by_sequence[(seq_id, strand)].append(score)
                    positions_by_sequence[(seq_id, strand)].append(position)
                batched_instances = []
                batched_positions = []

    if len(batched_instances) > 0:
        X = torch.cat(batched_instances).to(args.device)
        scores = torch.exp(model(X)[:,1]).detach().cpu().numpy()
        for i, (seq_id, position, strand) in enumerate(batched_positions):
            score = scores[i]
            if score < args.cutoff:
                continue
            if (seq_id, strand) not in scores_by_sequence:
                scores_by_sequence[(seq_id, strand)] = []
                positions_by_sequence[(seq_id, strand)] = []
            scores_by_sequence[(seq_id, strand)].append(score)
            positions_by_sequence[(seq_id, strand)].append(position)

    
    logger.info("processing the predictions ...")
    logger.info(f"results will be saved to {args.output} ...")
    fout = open(args.output,"w")

    for seq_id, strand in scores_by_sequence:
        scores = scores_by_sequence[(seq_id, strand)]
        positions = positions_by_sequence[(seq_id, strand)]
        signal = np.zeros(max(positions)+1)
        for p, s in zip(positions, scores):
            signal[p] = s
        if args.smooth > 1:
            signal = convolve(signal,np.ones(args.smooth)/args.smooth,mode="same")
        last_score, last_position = -1, -1
        next_score = signal[0]
        for position in range(signal.shape[0]-1):
            next_position = position + 1
            score = next_score
            next_score = signal[next_position]
            if (score > next_score) and (score > last_score) and (score > args.cutoff):
                if strand == "+":
                    print(seq_id, position + args.length - args.offset, score, strand, sep="\t", file=fout)
                else:
                    print(seq_id, position + args.offset, score, strand, sep="\t", file=fout)
            last_position = position
            last_score = score

    fout.close()
    logger.info("all done .")

                
if __name__ == "__main__":
    main()
