#!usr/bin/env python
from ushuffle import shuffle

def main():
    fout = open("examples/TPP.shuffled.fa","w")
    with open("examples/TPP.fa") as f:
        for header in f:
            sequence = next(f).strip()
            fout.write(header)
            fout.write(sequence + "\n")
            header = header.strip() + ":shuffled\n"
            fout.write(header)
            sequence = shuffle(sequence.encode(),1).decode()
            fout.write(sequence + "\n")
    fout.close()

if __name__ == "__main__":
    main()
