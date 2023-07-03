from pyfaidx import Fasta
import pandas as pd
import requests
import random
import shutil
import math
import gzip
import os

response = requests.get('https://ftp.ncbi.nlm.nih.gov/refseq/release/viral/viral.1.1.genomic.fna.gz', stream=True)
directory = "../data/viral"
filename = "viral.1.1.genomic.fna"
file_path =  directory + "/" + filename

if not os.path.exists(directory):
    os.makedirs(directory)

if response.status_code == 200:
    with gzip.open(response.raw, 'rb') as f_in:
        with open(file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

print("Downloaded viral genomes dataset!")

seqs = Fasta(str(file_path))


# split data into 90% train, 5% validation, 5% test

def shuffle_list(input_list, seed=2023):
    random.seed(seed)
    list_copy = input_list.copy()
    random.shuffle(list_copy)
    return list_copy

shuffled_seqs = shuffle_list(list(seqs.keys()))
train_part = int(len(shuffled_seqs)*0.9)
val_part = int(len(shuffled_seqs)*0.95)

train_seqs, val_seqs, test_seqs = (
    shuffled_seqs[:train_part], 
    shuffled_seqs[train_part:val_part], 
    shuffled_seqs[val_part:]
)


def process_seq(seq, seq_name, max_length=32768):
    """
    if sequence is longer than max_length,
    - divide sequence length by max_length to get number of splits
    - split sequence evenly into number of splits of size max_length
    """
    seq = str(seq)
    if len(seq) <= max_length:
        return {seq_name: seq}

    num_splits = len(seq) // max_length
    split_interval = (len(seq) - max_length) // num_splits
    split_starts = [0] + [split_interval*(i+1) for i in range(num_splits)]
    
    split_seqs = [seq[start:start+max_length] for start in split_starts]

    processed_seqs_map = {}
    for i, split in enumerate(split_seqs):
        split_name = seq_name + ".s" + str(i)
        processed_seqs_map[split_name] = split

    return processed_seqs_map


processed_seqs = {}
ds_splits = {"train": [], "valid": [], "test": []}

for seq_name in train_seqs:
    seq = seqs[seq_name]
    current_processed_seqs = process_seq(seq, seq_name)
    processed_seqs.update(current_processed_seqs)
    ds_splits["train"].extend(list(current_processed_seqs.keys()))

for seq_name in val_seqs:
    seq = seqs[seq_name]
    current_processed_seqs = process_seq(seq, seq_name)
    processed_seqs.update(current_processed_seqs)
    ds_splits["valid"].extend(list(current_processed_seqs.keys()))

for seq_name in test_seqs:
    seq = seqs[seq_name]
    current_processed_seqs = process_seq(seq, seq_name)
    processed_seqs.update(current_processed_seqs)
    ds_splits["test"].extend(list(current_processed_seqs.keys()))


# save processed seqs to fasta file
def write_fasta(file_name, processed_seqs):
    with open(file_name, 'w') as file:
        for k, v in processed_seqs.items():
            file.write('>' + k + '\n')
            file.write(v + '\n')

write_fasta(directory + "/" + "viral.processed.fa", processed_seqs)

# save ds_splits as bed file
chr_name = ds_splits['train'] + ds_splits['valid'] + ds_splits['test']
split = ['train']*len(ds_splits['train']) + \
        ['valid']*len(ds_splits['valid']) + \
        ['test']*len(ds_splits['test'])

ds_splits_df = pd.DataFrame(
    data={'chr_name': chr_name, 'start': ['']*len(chr_name), 'end': ['']*len(chr_name), 'split': split}, 
    columns=['chr_name', 'start', 'end', 'split']
)

ds_splits_df.to_csv(directory + "/" + "viral.processed.bed", sep="\t", index=False)

print("Created processed viral genomes dataset!")