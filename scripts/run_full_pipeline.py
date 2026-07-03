"""Run the full pipeline: download data, process all subjects, save results."""
import os
import json
import pandas as pd
from tqdm import tqdm
from src.preprocessing import preprocess_subject
from src.microstate_extraction import extract_microstates
from src.entropy_rate import compute_all_metrics


def main():
    DATASET_ROOT = './data_mci_ad'
    RESULTS_DIR = 'results'
    os.makedirs(RESULTS_DIR, exist_ok=True)  # was missing — to_csv() below would crash without it

    if not os.path.exists(DATASET_ROOT):
        os.system(f'aws s3 sync --no-sign-request s3://openneuro.org/ds004504 {DATASET_ROOT}')

    participants = pd.read_csv(f'{DATASET_ROOT}/participants.tsv', sep='\t')
    K = 4
    results = []

    for _, row in tqdm(participants.iterrows(), total=len(participants)):
        subj = row['participant_id']
        group_letter = row['Group']
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
            metrics = compute_all_metrics(labels, K=K)
            results.append({
                'subject_id': subj,
                'group': group,
                'ACE': metrics['ACE'],
                'H_rate': metrics['H_rate'],
                'H_occupancy': metrics['H_occupancy'],
                'row_entropy': json.dumps(metrics['row_entropy'].tolist()),
            })
        except Exception as e:
            print(f"Error {subj}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, 'entropy_rate_results.csv'), index=False)
    print(df.groupby('group')['ACE'].describe())


if __name__ == '__main__':
    main()
