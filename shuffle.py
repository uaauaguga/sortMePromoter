#!usr/bin/env python
from ushuffle import shuffle

def main():
    fout = open("examples/RF00174-Cobalamin-U75-D25.fa","w")
    with open("examples/RF00174-Cobalamin-U75-D25.pos.fa") as f:
        for header in f:
            sequence = next(f).strip()
            fout.write(header)
            fout.write(sequence + "\n")
            header = header.strip() + ":shuffled\n"
            fout.write(header)
            sequence = shuffle(sequence.encode(),1).decode()
            fout.write(sequence + "\n")
    fout.close()

main()
