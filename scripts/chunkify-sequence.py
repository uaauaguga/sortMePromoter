#!/usr/bin/env python
import argparse
from collections import defaultdict
from tqdm import tqdm
import os
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
logger = logging.getLogger("groupping sequences")

def main():
    parser = argparse.ArgumentParser(description='groupping input sequence by prespecified chunk size')
    parser.add_argument('--input', '-i', type=str, required=True, help='input sequence in fasta format')
    parser.add_argument('--output-directory','-od', type=str ,required=True, help="output groupped sequences")
    parser.add_argument('--chunk-size','-cs',type=int,default=200000,help="chunk size to use")
    args = parser.parse_args()
    
    logger.info("chunkify input sequences ...")
    if not os.path.exists(args.output_directory):
        logger.info(f"create output directory {args.output_directory} ...")
        os.mkdir(args.output_directory)

    i = 0
    n = 0
    fout = None
    with open(args.input) as f:
        for line in f:
            if line.startswith(">"):
                if n == 0:
                    if fout is not None:
                        fout.close()
                    index = str(i).zfill(4)
                    logger.info(f"start saving to chunk {index} ...")
                    fout = open(os.path.join(args.output_directory,f"{index}.fa"),"w")
                n += 1
            if n >= args.chunk_size:
                i += 1
                n = 0
            fout.write(line)
    logger.info("all done .")
        
    


if __name__ == "__main__":
    main()

