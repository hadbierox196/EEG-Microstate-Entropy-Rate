"""Run the full pipeline: download data, process all subjects, save results."""
import os
import pandas as pd
from tqdm import tqdm
from src.preprocessing import preprocess_subject
from src.microstate_extraction import extract_microstates
from src.entropy_rate import compute_maxent_entropy_rate
import numpy as np

def main():
    # Setup
    DATASET_ROOT = './data_mci_ad'
    if not os.path.exists(DATASET_ROOT):
        # Download using AWS CLI (or use openneuro-py)
        os.system('aws s3 sync --no-sign-request s3://openneuro.org/ds004504 ./data_mci_ad')
    
    participants = pd.read_csv(f'{DATASET_ROOT}/participants.tsv', sep='\t')
    K = 4
    results = []
    
    for _, row in tqdm(participants.iterrows(), total=len(participants)):
        subj = row['participant_id']
        group_letter = row['Group']
        # Map group
        if group_letter == 'C':
            group = 'HC'
        elif group_letter == 'A':
            group = 'AD'
        elif group_letter == 'F':
            group = 'FTD'
        else:
            continue
        eeg_path = os.path.join(DATASET_ROOT, subj, 'eeg',
                                f'{subj}_task-eyesclosed_eeg.set')
        if not os.path.exists(eeg_path):
            continue
        try:
            raw = preprocess_subject(eeg_path, file_format='set')
            labels, _ = extract_microstates(raw, K=K)
            H, T, pi = compute_maxent_entropy_rate(labels, K=K)
            results.append({'subject_id': subj, 'group': group,
                            'H': H})
        except Exception as e:
            print(f"Error {subj}: {e}")
    
    df = pd.DataFrame(results)
    df.to_csv('results/entropy_rate_results.csv', index=False)
    print(df.groupby('group')['H'].describe())

if __name__ == '__main__':
    main()
