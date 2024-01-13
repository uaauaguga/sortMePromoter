# sortMePromoter

## Introduction
- sortMePromoter is a tool for bacteria promoter prediction. It is simply a CNN with 10 resnet block
- It was trained to distinguish upstream 50nt of TSS identified by differential RNA-seq (dRNA-seq) from random genomic 50-mer, using data from diverse species
- sortMePromoter depends on pytorch

## Usage

```bash
scripts/sortMePromoter.py --fasta examples/TPP.with.shuffled.fa --output examples/TPP.with.shuffled.txt
``` 
 
