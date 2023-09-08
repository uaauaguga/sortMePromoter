#!/usr/bin/env python
import argparse
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, roc_curve
import numpy as np
def main():
    parser = argparse.ArgumentParser(description='performance evaluation')
    parser.add_argument('--input', '-i', type=str, required = True, help="input predicted TSS sites")
    #parser.add_argument('--output','-o', type=str, required = True, help="output performance")
    args = parser.parse_args()
    
    scores = []
    labels = []
    scores_by_sequence = {}
    labels_by_sequence = {}
    positions_by_sequence = {}
    seq_ids = set()
    with open(args.input) as f:
        for line in f:
            seq_id, position, score = line.strip().split("\t")[:3]
            seq_ids.add(seq_id)
            ref = int(seq_id.split(":")[-1])
            position, score = int(position), float(score)
            if abs(position-ref) <= 15:
                label = 1
            else:
                label = 0
            scores.append(score)
            labels.append(label)
            if seq_id not in scores_by_sequence:
                scores_by_sequence[seq_id] = []
                labels_by_sequence[seq_id] = []
                positions_by_sequence[seq_id] = []
            scores_by_sequence[seq_id].append(score)
            labels_by_sequence[seq_id].append(label)
            positions_by_sequence[seq_id].append(position)

    counts = [0, 0]
    for seq_id in scores_by_sequence:
        ss, ls = np.array(scores_by_sequence[seq_id]), np.array(labels_by_sequence[seq_id])
        i = np.argmax(ss)
        if ss[i] < 0.9:
            continue
        counts[ls[i]] += 1
        #print(seq_id, positions_by_sequence[seq_id][i], ss[i], ls[i], sep="\t")
    counts = np.array(counts)
    print(counts/counts.sum())



if __name__ == "__main__":
    main()
