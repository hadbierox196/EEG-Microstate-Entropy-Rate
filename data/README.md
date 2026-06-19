This directory should contain the OpenNeuro ds004504 dataset.

**Do not commit the data to GitHub** – it is large.
Instead, run `scripts/download_data.py` or use the AWS S3 sync command:

aws s3 sync --no-sign-request s3://openneuro.org/ds004504 ./data_mci_ad

The pipeline will automatically download it if not found.
