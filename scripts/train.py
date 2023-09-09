#!/usr/bin/env python
import os
import torch
from torch.optim import Adam
from dataset import PromoterSet,collate_fn
from torch.utils.data import DataLoader, SequentialSampler,RandomSampler
from torch.functional import F
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, roc_curve
from model import CNNClassifier
import argparse
import pandas as pd
from scipy.interpolate import interp1d
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger('train promoter predicter')

def main():
    parser = argparse.ArgumentParser(description='Train Promoter Predictor')
    parser.add_argument('--device','-d',default="cuda:1",choices=["cuda:0","cuda:1","cpu"],help="Device to run the model")
    parser.add_argument('--train-positive','-tp',required=True,help="positive sequence for training")
    parser.add_argument('--train-negative','-tn',required=True,help="negative sequence for training")
    parser.add_argument('--val-positive','-vp',required=True,help="positive sequence for evaluation")
    parser.add_argument('--val-negative','-vn',required=True,help="negative sequence for evaluation")
    parser.add_argument('--models','-m',required=True,help="directory to save models")
    parser.add_argument('--n-channels', '-c', type=int, default=64, help="number of channels to use")
    parser.add_argument('--n-blocks', '-b', type=int, default=10, help="number of resnet blocks to use")
    parser.add_argument('--kernel-size', '-k', type=int, default=5, help="convolution kernel size in the res-block")
    parser.add_argument('--length', '-l', type=int, default=50, help="length of the sequence set")
    parser.add_argument('--positive-fraction', '-pf', type=float, default=0.5, help="positive fraction")
    parser.add_argument('--shuffled-fraction', '-sf', type=float, default=0.0, help="shuffled sequence fraction in negative sequence")
    parser.add_argument('--batch-size', '-bs', type=int, default=1024, help="batch size for ttraining and evaluation")
    parser.add_argument('--unstratify', '-us', action = "store_true",  help="whether stratify the sample when sampling")
    args = parser.parse_args()

    if not os.path.exists(args.models):
        os.mkdir(args.models)

    train_set = PromoterSet(args.train_positive, args.train_negative, 
                           length=args.length, positive_fraction=args.positive_fraction, 
                           shuffled_fraction = args.shuffled_fraction, stratified = not args.unstratify)
    train_sampler = RandomSampler(train_set)
    train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=args.batch_size, collate_fn=collate_fn)

    val_set = PromoterSet(args.val_positive, args.val_negative, length=args.length, positive_fraction=args.positive_fraction, shuffled_fraction = 0)
    val_sampler = RandomSampler(val_set)
    val_loader = DataLoader(val_set, sampler=val_sampler, batch_size=args.batch_size, collate_fn=collate_fn)

    model = CNNClassifier(n_channels=args.n_channels, kernel_size = args.kernel_size, n_blocks = args.n_blocks)

    model = model.to(args.device)
    optimizer = Adam(model.parameters(), lr=0.001)

    criterion = torch.nn.NLLLoss()

    train_losses = []
    for e in range(50):
        i = 0
        for sequences, labels, group_ids in train_loader:
            #logger.info(f"run inference {i} ...")
            sequences, labels = sequences.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            pred = model(sequences)
            accuracy = (pred.argmax(axis=1)==labels).sum()/labels.shape[0]
            loss = criterion(pred,labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            if i%256 == 0:
                model = model.eval()
                logger.info(f"{e}-{i} train: {np.mean(train_losses)} {accuracy.item()}")
                train_losses = []
                records = []
                correct, N = 0, 0
                losses = []
                j = 0
                for sequences, labels, group_ids in val_loader:
                    if j > 32:
                        break
                    sequences, labels = sequences.to(args.device), labels.to(args.device)
                    pred = model(sequences)
                    loss = criterion(pred,labels)
                    losses.append(loss.item())
                    pred_proba = torch.softmax(pred,-1)[:,1].cpu().detach().numpy()
                    y_true = labels.cpu().detach().numpy()
                    for p, t, g in zip(pred_proba,y_true, group_ids):
                        records.append((p,t,g))
                    j += 1
                    correct += ((pred_proba>0.5).astype(int)==y_true).sum()
                    N  += y_true.shape[0]
                table = pd.DataFrame.from_records(records)
                table.columns = ["y pred","y true", "group ids"]
                #accuracy = table.groupby(["group ids"]).apply(lambda x:((x["y pred"]>0.5).astype(int)==x["y true"]).sum()/x.shape[0])
                #print(accuracy.sort_values(ascending=False))
                AUROC = roc_auc_score(table["y true"],table["y pred"])
                fpr, tpr, thresholds = roc_curve(table["y true"], table["y pred"])
                recall05 = round(float(interp1d(fpr,tpr)(0.05)),3)
                precision, recall, thresholds = precision_recall_curve(table["y true"],table["y pred"])
                AUPRC = auc(recall,precision)
                logger.info(f"{e}-{i} val: pooled {np.mean(losses)} {correct/N} {AUPRC} {AUROC} {recall05}")
                for group_id in table["group ids"].unique():
                    sub_table = table[table["group ids"]==group_id]
                    AUROC = roc_auc_score(sub_table["y true"],sub_table["y pred"])
                    fpr, tpr, thresholds = roc_curve(sub_table["y true"], sub_table["y pred"])
                    recall05 = round(float(interp1d(fpr,tpr)(0.05)),3)
                    precision, recall, thresholds = precision_recall_curve(sub_table["y true"],sub_table["y pred"])
                    AUPRC = auc(recall, precision)
                    logger.info(f"{e}-{i} val: {group_id} {AUPRC} {AUROC} {recall05}")
                model = model.train()
            i += 1
        logger.info(f"saving model at epoch {e} ...")
        torch.save(model.state_dict(),f"{args.models}/{e}.pt")
if __name__ == "__main__":
    main()
