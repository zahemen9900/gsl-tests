# GSL Tests

This repository contains pipelines, training code, and docs for Sign2Text and Text2Sign modeling, plus frontend/backend prototypes for serving models.

## Structure
- vm_scripts/: VM-ready scripts for preprocessing, text augmentation, and training (Sign2Text and Text2Sign).
- docs/: Runbooks, architecture notes, and deliverables; includes GCP VM guides and objective docs.
- backend/: FastAPI backend scaffolding and model service code.
- frontend/: Vite/TypeScript frontend scaffold (index page and src components).
- features/: Pose feature .npy samples used for development/debugging.
- sample_dataset/: Small sample dataset with videos and metadata for local testing.
- runs/: Saved checkpoints, embeddings, and training history from prior experiments.
- TEXT-TO-SIGN/: Notebooks and docs focused on Text2Sign objectives.
- SIGN-TO-TEXT/: Notebooks and docs focused on Sign2Text objectives.
- proc/: Processing outputs (metadata, stats, plots) when run locally; mirrored to GCS as needed.
- deliverables.md and docs/deliverables/: High-level milestone tracking and detailed deliverable notes.

## Pipelines
- Preprocessing: vm_scripts/preprocess_vm.py (Sign2Text), vm_scripts/preprocess_vm_t2s.py (Text2Sign).
- Text augmentation only: vm_scripts/text-aug-sign2text.py.
- Training: vm_scripts/training_vm_sign2text.py (seq2seq), vm_scripts/training_vm_text2sign.py (GAN-NAT).

## Buckets
Default bucket layout uses `gs://ghsl-model-artifacts/` with per-task subfolders (sign2text/text2sign under features/, proc/, runs/). Configure URIs via script flags if needed.
