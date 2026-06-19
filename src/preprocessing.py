"""Preprocessing functions for EEG data."""
import mne
import warnings
warnings.filterwarnings('ignore')

def preprocess_subject(raw_path, file_format='auto'):
    """
    Load, filter, rereference, and clean one subject's resting-state EEG.
    """
    # Load
    if file_format == 'set':
        raw = mne.io.read_raw_eeglab(raw_path, preload=True, verbose=False)
    elif file_format == 'edf':
        raw = mne.io.read_raw_edf(raw_path, preload=True, verbose=False)
    else:
        raw = mne.io.read_raw(raw_path, preload=True, verbose=False)

    raw.pick_types(eeg=True, eog=False, stim=False)
    raw.filter(l_freq=1.0, h_freq=40.0, method='iir', verbose=False)
    raw.set_eeg_reference('average', projection=False, verbose=False)

    # ICA
    ica = mne.preprocessing.ICA(
        n_components=0.95,
        method='fastica',
        random_state=42,
        max_iter=800,
        verbose=False
    )
    ica.fit(raw, verbose=False)
    try:
        eog_indices, _ = ica.find_bads_eog(raw, verbose=False)
        ica.exclude = eog_indices[:2]
    except Exception:
        pass
    raw_clean = raw.copy()
    ica.apply(raw_clean, verbose=False)
    return raw_clean
