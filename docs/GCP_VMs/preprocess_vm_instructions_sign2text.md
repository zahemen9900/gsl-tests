# GCE VM runbook for preprocessing pipeline

This guides you through setting up a Compute Engine VM and running the end-to-end preprocessing script (now located in `vm_scripts/`) that replaces the Colab notebook. Visuals are exported to disk instead of shown interactively.

## 1) VM creation (recommended)

- Machine: `e2-standard-16` (16 vCPU, 64 GB RAM) or larger if you want faster throughput.
- OS image: Debian 12 or Ubuntu 22.04.
- Boot disk: 200 GB (For decent I/O Operations)
- Additional persistent disk: 1 TB (balanced) mounted at `/mnt/disks/data`.
- Service account: one with Storage Object Admin (for GCS read/write).
- Allow default access scopes for Storage.

Example (replace PROJECT, ZONE). Setting a device name avoids guessing the by-id path:

```sh
gcloud compute disks create preprocessing-data-disk --size=1024GB --type=pd-balanced --project=PROJECT --zone=ZONE

gcloud compute instances create preprocessing-vm \
  --project=PROJECT --zone=ZONE \
  --machine-type=e2-standard-16 \
  --boot-disk-size=200GB \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --create-disk=auto-delete=yes,device-name=preprocessing-data-disk,name=preprocessing-data-disk,mode=rw,boot=no
```

SSH the VM into local machine terminal (based on current setup):

```sh
gcloud compute ssh --zone "africa-south1-a" "preprocessing-vm" --project "even-ally-480821-f3"
```

After the VM is up, attach and mount the disk:

```sh
sudo mkdir -p /mnt/disks/data
sudo mkfs.ext4 -F /dev/disk/by-id/google-preprocessing-data-disk
sudo mount /dev/disk/by-id/google-preprocessing-data-disk /mnt/disks/data
sudo chmod a+w /mnt/disks/data
```

Add to fstab to survive reboots:

```sh
echo "/dev/disk/by-id/google-preprocessing-data-disk /mnt/disks/data ext4 defaults 0 0" | sudo tee -a /etc/fstab
```

## 2) System deps

```sh
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip ffmpeg libgl1
```

## 3) Repo + venv

```sh
cd /mnt/disks/data
git clone https://github.com/zahemen9900/gsl-tests.git
cd gsl-tests
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install mediapipe opencv-python-headless google-generativeai pandas numpy tqdm matplotlib
```

## 4) (Optional) Gemini API keys

Create `/mnt/disks/data/gsl-tests/.env` with one key per line in `KEY_NAME=VALUE` format. Provide at least two keys for parallelism.
Example:

```env
GEMINI_KEY_1=AIza...foo
GEMINI_KEY_2=AIza...bar
```

## 5) Run the Sign2Text preprocessing pipeline (vm_scripts/preprocess_vm.py)

Key paths/URIs you may want to change:

- Input dataset GCS URI: `gs://ghsl-datasets/sample_dataset`
- Local workdir: `/mnt/disks/data` (holds downloads, features, proc, plots)
- Output GCS URI (optional upload): defaults to `gs://ghsl-model-artifacts/sign2text` (features → `.../features/pose_data`, proc → `.../proc`)
- Face downsample: default 5
- Keep low-quality clips: add `--keep-low-quality` to keep instead of dropping
- Enable text augmentation: add `--augment-text --env-path .env`

Base run (no text augmentation):

```sh
source /mnt/disks/data/gsl-tests/.venv/bin/activate
cd /mnt/disks/data/gsl-tests/vm_scripts
python preprocess_vm.py \
  --dataset-gcs-uri gs://ghsl-datasets/sample_dataset \
  --workdir /mnt/disks/data \
  --output-gcs-uri gs://ghsl-model-artifacts/sign2text \
  --face-downsample 5
```

With text augmentation:

```sh
python preprocess_vm.py \
  --dataset-gcs-uri gs://ghsl-datasets/sample_dataset \
  --workdir /mnt/disks/data \
  --output-gcs-uri gs://ghsl-model-artifacts/sign2text \
  --augment-text \
  --env-path .env
```

## 6) Outputs

- Features: `/mnt/disks/data/features/pose_data/*.npy`
- Metadata: `/mnt/disks/data/proc/processed_metadata.csv` (includes `feature_path` rewritten to GCS if `--output-gcs-uri` is set, and `feature_path_local` for the on-disk path)
- Debug CSVs: `/mnt/disks/data/proc/{failed_videos,low_quality_videos,missing_videos}.csv`
- Global stats: `/mnt/disks/data/proc/global_stats.npz`
- Plots (exported visuals): `/mnt/disks/data/proc/plots/frames_distribution.png`
- Optional text variations cache: `/mnt/disks/data/proc/sentence_variations.json`
- If `--output-gcs-uri` is set, the script copies features to `gs://ghsl-model-artifacts/sign2text/features/pose_data` and proc to `gs://ghsl-model-artifacts/sign2text/proc`.

## 7) Notes

- Reruns overwrite existing outputs; remove or rename previous runs if needed.
- Use `--max-workers` to cap CPU usage. Default is all cores.
- Ensure `gcloud` is authenticated on the VM (`gcloud auth login` or service account on the instance).
- The script exports visuals to disk and does not open any interactive windows.
