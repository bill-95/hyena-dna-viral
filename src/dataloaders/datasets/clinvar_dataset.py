import pickle
from random import random
import numpy as np
from pathlib import Path
from pyfaidx import Fasta
import torch

# helper functions
def exists(val):
    return val is not None

def coin_flip():
    return random() > 0.5


string_complement_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'a': 't', 'c': 'g', 'g': 'c', 't': 'a'}
# augmentation
def string_reverse_complement(seq):
    rev_comp = ''
    for base in seq[::-1]:
        if base in string_complement_map:
            rev_comp += string_complement_map[base]
        # if bp not complement map, use the same bp
        else:
            rev_comp += base
    return rev_comp


class SampleSeq():
    def __init__(
        self,
        fasta_file,
    ):
        fasta_file = Path(fasta_file)
        assert fasta_file.exists(), 'path to fasta file must exist'

        self.g = Fasta(str(fasta_file))

    def __call__(self, chrom, pos, ref, alt, variant_type, length=1024):
        # shift starting position 1 to the left for consistency with reference genome
        pos = pos - 1

        # check if the length of the sequence is even, if not add one base to the left
        left_length = length // 2
        if length % 2 != 0:
            left_length += 1

        # get the right length of the sequence
        right_length = length - left_length

        # get the sequence from the genome
        original_seq = str(self.g[str(chrom)][pos - left_length : pos + right_length + len(ref)])
        original_seq = original_seq.upper()

        # create the modified sequence
        modified_seq = original_seq[:left_length] + alt + original_seq[left_length+len(ref):]

        # trim ends of modified sequences to length
        trim_length = len(modified_seq) - length
        if trim_length % 2 == 0:
            start = trim_length // 2
            end = -start
        else:
            start = trim_length // 2
            end = -(start+1)

        return modified_seq[start:end]



class ClinvarDataset(torch.utils.data.Dataset):
    '''
    Creates a pytorch dataset based on provided sample_ids
    '''

    def __init__(
        self,
        samples_path,
        genome_path,
        split,
        max_length,
        d_output=2, # default binary classification
        tokenizer=None,
        tokenizer_name=None,
        use_padding=None,
        add_eos=False,
        rc_aug=False,
        return_augs=False,
    ):
        # read in sample data
        with open(f'{samples_path}/samples.p', 'rb') as f:
            self.samples = pickle.load(f)
            
        with open(f'{samples_path}/sample_splits.p', 'rb') as f:
            sample_splits = pickle.load(f)
            self.sample_ids = sample_splits[split]
            
        self.sample_seq = SampleSeq(genome_path)
        
        self.max_length = max_length
        self.use_padding = use_padding
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.return_augs = return_augs
        self.add_eos = add_eos
        self.d_output = d_output  # needed for decoder to grab
        self.rc_aug = rc_aug

        self.label_mapping = {
            "Pathogenic": 1,
            "Likely pathogenic": 1,
            "Benign": 0,
            "Likely benign": 0
        }

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        sample = self.samples[sample_id]
        x = self.sample_seq(*sample[:-1], length=self.max_length)
        y = self.label_mapping[sample[-1]]

        # apply rc_aug here if using
        if self.rc_aug and coin_flip():
            x = string_reverse_complement(x)

        seq = self.tokenizer(x,
            add_special_tokens=False,
            padding="max_length" if self.use_padding else None,
            max_length=self.max_length,
            truncation=True,
        )  # add cls and eos token (+2)
        seq = seq["input_ids"]  # get input_ids

        # need to handle eos here
        if self.add_eos:
            # append list seems to be faster than append tensor
            seq = seq + [self.tokenizer.sep_token_id]
            # seq.append(self.tokenizer.sep_token_id)

        # convert to tensor
        seq = torch.LongTensor(seq)

        # need to wrap in list
        target = torch.LongTensor([y])

        return seq, target