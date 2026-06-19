from pycrostates.cluster import ModKMeans
from pycrostates.preprocessing import extract_gfp_peaks

def extract_microstates(raw_clean, K=4):
    gfp_peaks = extract_gfp_peaks(raw_clean, picks='eeg')
    ModK = ModKMeans(n_clusters=K, random_state=42, n_init=100)
    ModK.fit(gfp_peaks, verbose=False)
    segmentation = ModK.predict(raw_clean, picks='eeg', verbose=False)
    labels = segmentation.labels
    return labels, ModK
