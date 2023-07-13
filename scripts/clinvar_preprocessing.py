import gzip
import os
import pickle
import requests
import random
import shutil
import pandas as pd
import tqdm
from sklearn.model_selection import train_test_split


response = requests.get('https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz', stream=True)
directory = "../data/clinvar"
filename = "variant_summary.txt"
file_path =  directory + "/" + filename

if not os.path.exists(directory):
    os.makedirs(directory)

if response.status_code == 200:
    with gzip.open(response.raw, 'rb') as f_in:
        with open(file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

print("Downloaded clinvar dataset!")

def load_data():
    df = pd.read_csv('../data/clinvar/variant_summary.txt', delimiter='\t')
    # use data from latest version of the human genome
    df = df[df.Assembly == 'GRCh38']
    # filter for relevant fields
    df = df[['#AlleleID', 'Type', 'ClinicalSignificance', 'Assembly', 'Chromosome', 'PositionVCF', 'ReferenceAlleleVCF', 'AlternateAlleleVCF']]
    # filter for only those that are "Pathogenic", or "Benign" to predict
    df = df[df.ClinicalSignificance.isin((
        "Pathogenic",
        "Benign"
    ))]
    df = df[df.PositionVCF != -1] # missing values
    df = df[df.ReferenceAlleleVCF != 'na'] # missing values
    df = df[df.AlternateAlleleVCF != 'na'] # missing values
    df = df[df.Type != 'Variation'] # unknown variant type
    df = df[df.Chromosome != 'MT'] # filter out mitochondrial variants

    return df

df = load_data()

print("Loaded clinvar dataset!")


# format samples here for training later
samples = {}

for i, row in tqdm.tqdm(df.iterrows()):
    sample_id = row['#AlleleID']
    chrom = row.Chromosome
    pos = row.PositionVCF
    ref = row.ReferenceAlleleVCF
    alt = row.AlternateAlleleVCF
    variant_type = row.Type
    label = row.ClinicalSignificance

    samples[sample_id] = (chrom, pos, ref, alt, variant_type, label)
    
    
# create splits
sample_ids = list(samples.keys())
train_ids, test_val_ids = train_test_split(sample_ids, test_size=0.2, random_state=42)
val_ids, test_ids = train_test_split(test_val_ids, test_size=0.5, random_state=42)


# saving processed training samples
with open('../data/clinvar/samples.p', 'wb') as f:
    pickle.dump(samples, f)

with open('../data/clinvar/sample_splits.p', 'wb') as f:
    pickle.dump({
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }, f)