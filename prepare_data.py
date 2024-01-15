import pandas as pd
import numpy as np
import random 
from doublet_shuffle import doublet_shuffle

seed = 11
random.seed(seed)


def prepare_training(path):
    df = pd.read_csv(path+'/train.csv', header=None)
    num_seqs = len(df)

    df.columns = ['sequence']
    df['label'] = [1]*num_seqs

    seqs = list(df['sequence'])
    max_len = len(max(seqs, key=len))
    min_len = len(min(seqs, key=len))

    # generate dinucleotide-preserving shuffles (negative sequences)
    for i in range(num_seqs):
        dinuc_shuffle = doublet_shuffle(df.loc[i, 'sequence'])
        df = df.append({'sequence': dinuc_shuffle, 'label': 0}, ignore_index=True)

    # pad sequences with 'N'
    for i, row in df.iterrows():
        df.loc[i, 'sequence'] = row['sequence'] + (max_len-len(row['sequence'])) * 'N'

    # shuffle dataset
    df = df.sample(frac=1, random_state=seed)
    df.to_csv(path+'/train_processed.csv', encoding='utf-8', index=False)


def prepare_testing(path):
    f = pd.read_csv(path, header=None)
    num_seqs = len(df)

    df.columns = ['sequence', 'label']

    seqs = list(df['sequence'])
    max_len = len(max(seqs, key=len))
    min_len = len(min(seqs, key=len))

    # pad sequences
    for i, row in df.iterrows():
        df.loc[i, 'sequence'] = row['sequence'] + (max_len-len(row['sequence'])) * 'N'

    return df


def main():
    prepare_training(path='data')


if __name__ == '__main__':
    main()

