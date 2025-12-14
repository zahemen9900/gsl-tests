# GPU VM runbook (training + preprocessing)

This runbook covers spinning up a GPU VM, setting up the environment, and running preprocessing and training for both Sign2Text and Text2Sign using the `vm_scripts/` folder and the `gs://ghsl-model-artifacts` bucket layout.

## 1) VM shape and disks

- GPU: `n1-standard-16` + `A100 40GB` (or `L4` for lower cost). Adjust machine type to feed the GPU (>=16 vCPU/64 GB RAM recommended for dataloader headroom).
- Boot disk: 200 GB (Ubuntu 22.04 LTS).
- Data disk: 1–2 TB balanced PD mounted at `/mnt/disks/data`.
- Service account: Storage Object Admin (read/write to `gs://ghsl-model-artifacts/*`).

Example (replace PROJECT, ZONE):

```sh
# Create data disk
gcloud compute disks create train-data-disk --size=1024GB --type=pd-balanced --project=PROJECT --zone=ZONE

# Create VM with GPU
gcloud compute instances create training-gpu \
  --project=PROJECT --zone=ZONE \
  --machine-type=n1-standard-16 \
  --accelerator=type=nvidia-tesla-a100,count=1 \
  --boot-disk-size=200GB \
  --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
  --maintenance-policy=TERMINATE --restart-on-failure \
  --service-account=SERVICE_ACCOUNT_EMAIL \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --create-disk=auto-delete=yes,device-name=train-data-disk,name=train-data-disk,mode=rw,boot=no
```

After the VM is up, attach/mount the disk:

```sh
sudo mkdir -p /mnt/disks/data
sudo mkfs.ext4 -F /dev/disk/by-id/google-train-data-disk
sudo mount /dev/disk/by-id/google-train-data-disk /mnt/disks/data
sudo chmod a+w /mnt/disks/data
# Persist across reboots
echo "/dev/disk/by-id/google-train-data-disk /mnt/disks/data ext4 defaults 0 0" | sudo tee -a /etc/fstab
```

## 2) System deps and NVIDIA stack

```sh
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip ffmpeg libgl1 gcc
# Install NVIDIA drivers + CUDA toolkit via apt (newer images often ship drivers already)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-4
# Reboot after driver install if needed
```

Verify:

```sh
nvidia-smi
```

## 3) Repo + venv + Python deps

```sh
cd /mnt/disks/data
git clone https://github.com/zahemen9900/gsl-tests.git
cd gsl-tests
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# Core deps (CPU + GPU build of torch)
pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install mediapipe opencv-python-headless pandas numpy tqdm matplotlib transformers
```

(Optional) Gemini augmentation keys: place in `.env` (KEY=VALUE per line) at repo root.

## 4) Bucket layout (model-artifacts)

- Sign2Text preprocess outputs: `gs://ghsl-model-artifacts/sign2text/features/pose_data/*.npy`, `gs://ghsl-model-artifacts/sign2text/proc/processed_metadata.csv`, `gs://ghsl-model-artifacts/sign2text/proc/global_stats.npz`.
- Text2Sign preprocess outputs: `gs://ghsl-model-artifacts/text2sign/features/text2sign_pose/*.npy`, `gs://ghsl-model-artifacts/text2sign/proc/text2sign_processed_metadata.csv`, `gs://ghsl-model-artifacts/text2sign/proc/text2sign_global_stats.npz`.
- Training runs: `gs://ghsl-model-artifacts/sign2text/runs` and `gs://ghsl-model-artifacts/text2sign/runs` (scripts stage locally then sync each epoch).

## 5) Preprocessing commands (run from vm_scripts)

Activate env and `cd`:

```sh
source /mnt/disks/data/gsl-tests/.venv/bin/activate
cd /mnt/disks/data/gsl-tests/vm_scripts
```

Sign2Text:

```sh
python preprocess_vm.py \
  --dataset-gcs-uri gs://ghsl-datasets/sample_dataset \
  --workdir /mnt/disks/data \
  --output-gcs-uri gs://ghsl-model-artifacts/sign2text \
  --face-downsample 5
```

Text2Sign:

```sh
python preprocess_vm_t2s.py \
  --dataset-gcs-uri gs://ghsl-datasets/text2sign_dataset \
  --workdir /mnt/disks/data \
  --output-gcs-uri gs://ghsl-model-artifacts/text2sign \
  --face-downsample 5
```

Notes:

- Metadata CSVs rewrite `feature_path` to the bucket and keep `feature_path_local` for on-disk paths.
- If you skip `--output-gcs-uri`, everything stays local under `/mnt/disks/data/{features,proc}`.

## 6) Training commands (GPU)

Stay in `vm_scripts/` with env active. The scripts will pull metadata/stats/features from GCS into a local cache (`/tmp/gcs_cache_*` by default) and sync runs back to the bucket each epoch.

Sign2Text (seq2seq):

```sh
python training_vm_sign2text.py \
  --processed-meta gs://ghsl-model-artifacts/sign2text/proc/processed_metadata.csv \
  --feature-dir gs://ghsl-model-artifacts/sign2text/features/pose_data \
  --global-stats gs://ghsl-model-artifacts/sign2text/proc/global_stats.npz \
  --out-dir gs://ghsl-model-artifacts/sign2text/runs \
  --batch-size 32 \
  --epochs 30 \
  --gcs-cache-dir /mnt/disks/data/cache_sign2text
```

Text2Sign (GAN-NAT):

```sh
python training_vm_text2sign.py \
  --processed-meta gs://ghsl-model-artifacts/text2sign/proc/text2sign_processed_metadata.csv \
  --feature-dir gs://ghsl-model-artifacts/text2sign/features/text2sign_pose \
  --global-stats gs://ghsl-model-artifacts/text2sign/proc/text2sign_global_stats.npz \
  --out-dir gs://ghsl-model-artifacts/text2sign/runs \
  --batch-size 16 \
  --epochs 50 \
  --gcs-cache-dir /mnt/disks/data/cache_text2sign
```

## 7) Quick checks

- GPU visible: `nvidia-smi`
- Torch sees CUDA: `python - <<'PY'
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY`
- Disk free: `df -h /mnt/disks/data`

## 8) Operational notes

- Runs sync to GCS each epoch; for large runs you can reduce sync frequency by copying at the end instead.
- Pre-warming the cache: `gcloud storage cp -r gs://ghsl-model-artifacts/sign2text/features/pose_data /mnt/disks/data/cache_sign2text/model-artifacts/sign2text/features/` (similarly for text2sign) if you want fewer on-the-fly downloads.
- If you use gcsfuse instead of per-file copies, mount under `/mnt/disks/data/gcs` and point `--feature-dir` and `--processed-meta` to the mounted paths.
- Increase `--num-workers` cautiously; too many workers can starve the GPU if I/O is slow.
- Logs and checkpoints live under the local run directory (`/tmp/gcs_cache_*/*/runs_*` when using GCS outputs). Monitor free space.
