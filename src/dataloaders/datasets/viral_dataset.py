from pathlib import Path
from pyfaidx import Fasta
import polars as pl
import pandas as pd
import torch
from random import randrange, random
import numpy as np


"""

Dataset for sampling arbitrary intervals from the viral genomes.

"""


# helper functions

def exists(val):
    return val is not None

def coin_flip():
    return random() > 0.5

# augmentations

string_complement_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'a': 't', 'c': 'g', 'g': 'c', 't': 'a'}

def string_reverse_complement(seq):
    rev_comp = ''
    for base in seq[::-1]:
        if base in string_complement_map:
            rev_comp += string_complement_map[base]
        # if bp not complement map, use the same bp
        else:
            rev_comp += base
    return rev_comp


class FastaInterval():
    def __init__(
        self,
        *,
        fasta_file,
        # max_length = None,
        return_seq_indices = False,
        shift_augs = None,
        rc_aug = False
    ):
        fasta_file = Path(fasta_file)
        assert fasta_file.exists(), 'path to fasta file must exist'

        self.seqs = Fasta(str(fasta_file))
        self.return_seq_indices = return_seq_indices
        # self.max_length = max_length # -1 for adding sos or eos token
        self.shift_augs = shift_augs
        self.rc_aug = rc_aug

        # calc len of each chromosome in fasta file, store in dict
        self.chr_lens = {}

        for chr_name in self.seqs.keys():
            # remove tail end, might be gibberish code
            # truncate_len = int(len(self.seqs[chr_name]) * 0.9)
            # self.chr_lens[chr_name] = truncate_len
            self.chr_lens[chr_name] = len(self.seqs[chr_name])


    def __call__(self, chr_name, return_augs = False):
        chromosome = self.seqs[chr_name]
        seq = str(chromosome)

        return seq

class ViralDataset(torch.utils.data.Dataset):

    '''
    Loop thru bed file, retrieve (chr, start, end), query fasta file for sequence.
    
    '''

    def __init__(
        self,
        split,
        bed_file,
        fasta_file,
        max_length,
        pad_max_length=None,
        tokenizer=None,
        tokenizer_name=None,
        add_eos=False,
        return_seq_indices=False,
        shift_augs=None,
        rc_aug=False,
        return_augs=False
    ):

        self.max_length = max_length
        self.pad_max_length = pad_max_length if pad_max_length is not None else max_length
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.return_augs = return_augs
        self.add_eos = add_eos

        bed_path = Path(bed_file)
        assert bed_path.exists(), 'path to .bed file must exist'

        # read bed file
        df_raw = pd.read_csv(str(bed_path), sep = '\t', names=['chr_name', 'start', 'end', 'split'])
        # select only split df
        self.df = df_raw[df_raw['split'] == split]

        self.fasta = FastaInterval(
            fasta_file = fasta_file,
            # max_length = max_length,
            return_seq_indices = return_seq_indices,
            shift_augs = shift_augs,
            rc_aug = rc_aug
        )

    def __len__(self):
        return len(self.df)

    def replace_value(self, x, old_value, new_value):
        return torch.where(x == old_value, new_value, x)

    def __getitem__(self, idx):
        """Returns a sequence of specified len"""
        # sample a random row from df
        row = self.df.iloc[idx]
        # row = (chr, start, end, split)
        chr_name, start, end = (row[0], row[1], row[2])

        seq = self.fasta(chr_name, return_augs=self.return_augs)

        if self.tokenizer_name == 'char':

            seq = self.tokenizer(seq,
                padding="max_length",
                max_length=self.pad_max_length,
                truncation=True,
                add_special_tokens=False)  # add cls and eos token (+2)

            seq = seq["input_ids"]  # get input_ids

            # need to handle eos here
            if self.add_eos:
                # append list seems to be faster than append tensor
                seq.append(self.tokenizer.sep_token_id)

        elif self.tokenizer_name == 'bpe':
            seq = self.tokenizer(seq, 
                # add_special_tokens=False, 
                padding="max_length",
                max_length=self.pad_max_length,
                truncation=True,
            ) 
            # get input_ids
            if self.add_eos:
                seq = seq["input_ids"][1:]  # remove the bos, keep the eos token
            else:
                seq = seq["input_ids"][1:-1]  # remove both special tokens
        
        # convert to tensor
        seq = torch.LongTensor(seq)  # hack, remove the initial cls tokens for now

        # replace N token with a pad token, so we can ignore it in the loss
        seq = self.replace_value(seq, self.tokenizer._vocab_str_to_int['N'], self.tokenizer.pad_token_id)

        data = seq[:-1].clone()  # remove eos
        target = seq[1:].clone()  # offset by 1, includes eos

        return data, target
