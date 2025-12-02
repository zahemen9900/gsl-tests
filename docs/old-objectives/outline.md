
GhSL Learning Platform — Model Specification & Database Schema
---

## Document overview

This canvas contains two connected sections to support the hackathon build and pitch:

1. **Model & System Specification** — detailed design for the embedding-based sentence model plus the simpler DTW/cosine similarity word dictionary flow, preprocessing, training, inference, evaluation, deployment, and integration with the web frontend.
    
2. **Postgres Database Schema & Storage Plan** — full schema design with table definitions, example `CREATE TABLE` statements, indices (including `pgvector` setup), and notes on what to store where. This covers lessons, videos, sentence metadata, embeddings, user progress and audit/logging.
    

---

# Part A — Model & System Specification

## Goal

Build a robust, hackathon-ready learning platform for Ghanaian Sign Language (GhSL).

Two parallel capabilities:

- **Word/Dictonary Mode** — fast, client-side matching of isolated signs using existing OpenPose keypoints and DTW/cosine similarity. This yields instant feedback for basic lessons.
    
- **Sentence Mode (Primary)** — embedding-based model trained on a subset of SignTalk-GH that maps pose/time sequences to compact embeddings; at inference we do nearest-neighbour retrieval against stored sentence embeddings to return candidate sentence(s) and confidence.
    

Both modes integrate into the same UI and share preprocessing and normalization so the frontend pipeline is unified.

---

## Overall architecture (high-level)

- **Frontend (Vite + React + Tailwind + shadcn)**
    
    - MediaPipe.js Holistic for real-time pose extraction.
        
    - Local smoothing & normalization module.
        
    - Word-mode local comparator (DTW/cosine) for isolated signs.
        
    - UI to capture sequence and call backend for sentence-mode inference.
        
- **Backend**
    
    - FastAPI (Python) serving ONNXRuntime model endpoint for encoder inference.
        
    - Supabase (Postgres) with `pgvector` extension to store embeddings & metadata.
        
    - Optional worker for batch preprocessing and embedding creation (CPU for MediaPipe, GPU for training).
        
- **Offline processing**
    
    - MediaPipe or OpenPose pipeline to convert SignTalk-GH videos → per-frame keypoints `.npy` / `.json`.
        
    - Subsample dataset to ~1–1.5k videos (balanced across categories and signers), then compute final embeddings for storage.
        

---

## Preprocessing pipeline (detailed)

1. **Video sampling & trimming**
    
    - Resample all clips to a canonical FPS (10–15 fps). For long clips, uniformly sample frames to fixed length `T` (e.g., 32 or 48). If a clip has sign-level temporal metadata, trim to region of interest.
        
2. **Pose extraction**
    
    - Use **MediaPipe Holistic** to extract body + hands + face keypoints per frame (x, y, z, confidence). Save as `.npy` arrays shaped `(T, K, 3)` or flattened `(T, F)` where `F = K*3`.
        
    - If OpenPose keypoints available, convert to the same key ordering to keep consistency.
        
3. **Normalization**
    
    - Translate coordinates so torso origin = 0 (e.g., mid-shoulder point).
        
    - Scale by torso length (distance between shoulders or shoulder-to-hip) to remove subject size variance.
        
    - Optionally rotate to canonical orientation (align shoulders horizontally) if needed.
        
4. **Derived features** (recommended)
    
    - Per-frame joint velocities (frame-to-frame deltas).
        
    - Selected joint angles or distances (e.g., wrist-to-shoulder, wrist-to-wrist).
        
    - Hand orientation proxies if available (palm normal approximations).
        
5. **Padding / masking**
    
    - Pad sequences shorter than `T` with zeros and pass a binary mask; truncate longer sequences by uniform sampling or center-cropping frames.
        
6. **Save**
    
    - Store processed sequences per video as compressed `.npz` or `.npy` files and maintain a CSV/metadata file mapping `video_id -> sentence_id -> path`.
        

---

## Model design (embedding encoder — recommended)

### Input & feature shape

- Input: `X` ∈ ℝ^{T × F} (T = 32–64 frames, F = flattened keypoint feature size after optional derived features).
    
- Example F: 33 body joints × 3 + 2 hands × 21 joints × 3 ≈ 198–216 raw features (exact number depends on chosen joint set). After projection, frame embedding dims reduce to `C` (e.g., 128).
    

### Architecture (balanced for RTX 3050)

- **FrameProjector (MLP)**: per frame, `F -> C` (two-layer MLP with ReLU). Produces `T × C` sequence.
    
- **Temporal Encoder** (one of):
    
    - **Option 1 — Bidirectional GRU**: 2 layers, hidden=128; mean pooling over time -> 256-d vector.
        
    - **Option 2 — Tiny Transformer Encoder**: 2–4 layers, 4 heads, hidden dim 128–192 (use fewer heads to reduce cost).
        
    - **Option 3 — TemporalConvNet (TCN)**: stack of 1D convs with residuals.
        
- **Embedding head**: Linear -> 256 dims -> L2 normalize to unit vector.
    

### Loss & training objective

- **Primary objective**: Contrastive InfoNCE or Triplet Loss using sentenceId as the grouping key. Because each sentence has multiple renditions (1A,1B...), positives are natural from dataset.
    
- **Alternative/classification**: If you construct grouped classes (e.g., collapse 4000 sentences into 500 thematic clusters) you can optionally add classification head + cross-entropy.
    

### Augmentations used during training

- Time warp (speed changes), frame drop, small rotation, coordinate jitter, horizontal flip (if sign semantics allow), scaling, and noise injection.
    

### Training hyperparams (starter)

- Batch size: 4–8 (adjust to avoid OOM)
    
- Sequence length: T=32 (experiment with 48)
    
- Optimizer: AdamW, lr=1e-4, weight_decay=1e-5
    
- Epochs: 20–50 (monitor val recall@k)
    
- Mixed precision: enable `torch.cuda.amp`
    

---

## Inference & retrieval

1. **Offline**: Encode all reference videos in chosen subset → embeddings `E_ref` ∈ ℝ^{N × d} (d=256). Store embeddings in PG vector store (pgvector) with an index (HNSW or IVF). Keep mapping `embedding_id -> video_id -> sentence_id`.
    
2. **Live**: User records or streams a sign clip (T frames). Client extracts keypoints -> normalizes -> sends compressed sequence to backend (or compute embedding in-browser if small model used). Backend runs encoder -> returns user embedding `e_q`.
    
3. **Search**: Query nearest neighbours in vector DB (k=5). Return top matches and their sentence metadata. Compute confidence as cosine similarity.
    
4. **Feedback**: Translate top-k results to UX — show the predicted sentence(s), provide per-frame similarity heatmap (optional), and give a text hint for corrective action.
    

---

## Word-mode: DTW / cosine similarity (fast path)

- For the isolated-word lexicon you already have (OpenPose), pre-store the canonical per-frame keypoint sequences.
    
- Local (client or lightweight server) algorithm:
    
    1. Capture user sequence (T_u) and canonical reference (T_r).
        
    2. Optionally resample to same length or use DTW to align.
        
    3. Compute DTW distance or cosine similarity on flattened or pooled features.
        
    4. Threshold for acceptance; provide micro-feedback (which frames misalign).
        
- _Performance:_ DTW cost grows with T^2 but lexicon size (≈1.2k) is manageable — can do per-lesson restricted searches only among candidate signs.
    

---

## Explainability & UX signals (critical for judges)

- **Top-k retrieval display** with similarity scores.
    
- **Frame-level heatmap** showing which frames matched strongly/weakly vs reference (use dot-product of frame embeddings or moving-window similarity).
    
- **Confidence thresholds** and adaptive prompts: e.g., "Rotate your palm outward" (based on which joints contributed to mismatch).
    

---

## Metrics to report (for pitch)

- Retrieval: **Recall@1**, **Recall@5**, **Mean Reciprocal Rank (MRR)**.
    
- Classification (if used): Top-1 / Top-5 accuracy.
    
- Latency: embedding inference time + ANN query time.
    
- Robustness: test on unseen signers (if possible) and show gap.
    

---

## Model export & serving

- Export trained encoder as **ONNX** for server-side inference via ONNXRuntime (lightweight) or as **TFLite** if you want on-device inference.
    
- Host FastAPI endpoint that accepts compressed pose sequences and returns top-k matches.
    
- Use asynchronous workers for batching requests if demo traffic spikes.
    

---

## Hackathon MVP scope & timeline

- Pre-hackathon (recommended): preprocess dataset subset, train encoder on subset, seed Supabase with embeddings & metadata, get frontend scaffold ready with MediaPipe.js.
    
- Hackathon timeline (36 hrs) — high-level:
    
    - H0–6: Frontend + MediaPipe integration + DB skeleton.
        
    - H6–18: Word-mode implemented locally (DTW) + demo-ready UX.
        
    - H18–28: Hook backend encoder endpoint & do embedding retrieval + top-k UI.
        
    - H28–36: Polish, prepare demo script and evaluation cards.
        

---

# Part B — Postgres Schema & Storage Plan (Supabase + pgvector)

> Note: assume Supabase project with `pgvector` extension enabled. Use `vector` type for embedding columns.

## High-level data objects

- `users` — app users (learners)
    
- `sentences` — canonical sentence entries (from SignTalk-GH metadata)
    
- `videos` — video files (multiple renditions per `sentence`) with pointers to storage (S3 or Supabase Storage)
    
- `video_keypoints` — preprocessed keypoint `.npy` references (path, shape)
    
- `embeddings` — embedding vectors (one per video or one per sentence-centroid)
    
- `lessons` — lesson definitions (sequence of activities)
    
- `attempts` — user attempt logs (for analytics & feedback)
    
- `progress` — user progression, XP, badges
    
- `audit_logs` — system events & preprocessing jobs
    

## Example SQL schema (Postgres + pgvector)

```sql
-- Enable extension (run once as superuser)
-- CREATE EXTENSION IF NOT EXISTS vector;

-- 1. users
CREATE TABLE users (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  email text UNIQUE,
  display_name text,
  created_at timestamptz DEFAULT now()
);

-- 2. sentences (canonical)
CREATE TABLE sentences (
  id serial PRIMARY KEY,
  sentence_id text NOT NULL, -- dataset id like '1'
  sentence_text text NOT NULL,
  category text,
  notes text,
  created_at timestamptz DEFAULT now()
);

-- 3. videos (many per sentence)
CREATE TABLE videos (
  id serial PRIMARY KEY,
  sentence_id int REFERENCES sentences(id) ON DELETE CASCADE,
  video_label text, -- e.g. '1A'
  storage_path text, -- e.g. 'gs://...' or Supabase path
  duration_secs real,
  fps int,
  created_at timestamptz DEFAULT now()
);

-- 4. video_keypoints
CREATE TABLE video_keypoints (
  id serial PRIMARY KEY,
  video_id int REFERENCES videos(id) ON DELETE CASCADE,
  keypoint_path text, -- path to compressed .npy/.npz in storage
  frames int,
  feature_dim int,
  created_at timestamptz DEFAULT now()
);

-- 5. embeddings (one per video or sentence centroid)
CREATE TABLE embeddings (
  id serial PRIMARY KEY,
  video_id int REFERENCES videos(id) ON DELETE CASCADE,
  embedding vector(256),
  created_at timestamptz DEFAULT now()
);
CREATE INDEX ON embeddings USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);

-- 6. lessons (Duolingo-style sequences)
CREATE TABLE lessons (
  id serial PRIMARY KEY,
  slug text UNIQUE,
  title text,
  description text,
  difficulty smallint,
  created_at timestamptz DEFAULT now()
);

-- 7. lesson_items (sequence of activities)
CREATE TABLE lesson_items (
  id serial PRIMARY KEY,
  lesson_id int REFERENCES lessons(id) ON DELETE CASCADE,
  position int,
  kind text, -- 'watch_choice', 'practice_sign', 'quiz'
  payload jsonb,
  created_at timestamptz DEFAULT now()
);

-- 8. attempts (user attempts on lesson items)
CREATE TABLE attempts (
  id serial PRIMARY KEY,
  user_id uuid REFERENCES users(id) ON DELETE CASCADE,
  lesson_item_id int REFERENCES lesson_items(id),
  timestamp timestamptz DEFAULT now(),
  success boolean,
  score real,
  raw_payload jsonb -- store user sequence or summarized features for audits
);

-- 9. progress & badges
CREATE TABLE progress (
  id serial PRIMARY KEY,
  user_id uuid REFERENCES users(id) ON DELETE CASCADE,
  lesson_id int REFERENCES lessons(id),
  xp int DEFAULT 0,
  last_completed timestamptz
);

-- 10. audit logs
CREATE TABLE audit_logs (
  id serial PRIMARY KEY,
  event_type text,
  payload jsonb,
  created_at timestamptz DEFAULT now()
);
```

---

## Notes & practical tips for DB

- **Embedding indexing**: For pgvector use `ivfflat` or `hnsw` (hnsw supported in newer setups) and tune `lists` parameter to trade recall vs latency.
    
- **Storage of keypoints**: Save `.npz` blobs in Supabase Storage and keep `keypoint_path` reference. Keypoint files are small and fast to serve for debug & re-encoding.
    
- **Thumbnail & video hosting**: Use Supabase Storage (S3 compatible) or a small GCS bucket; store URLs in `videos.storage_path`.
    
- **Backfill**: When you preprocess subset of SignTalk-GH, populate `sentences`, `videos`, `video_keypoints`, and `embeddings` atomically so front-end can query immediately.
    
- **Versioning**: Add `schema_version` or `dataset_version` columns if you plan iterative preprocessing.
    

---

## Example row flows (in app)

1. Admin uploads dataset subset → job creates `sentences` and `videos` rows → preprocessing writes `video_keypoints` and outputs embeddings → embeddings inserted into `embeddings` table with `video_id`.
    
2. User takes lesson → frontend uses MediaPipe to capture sequence → local library performs DTW for word-mode or calls encoder endpoint → encoder returns embedding → backend queries `embeddings` for top-k → backend returns sentence + metadata to frontend → `attempts` row saved for analytics.
    

---

# Appendix: Quick dev checklist

-  Spin Supabase project and enable `pgvector` extension.
    
-  Create storage bucket for keypoints and videos.
    
-  Preprocess a small subset (100–200) to validate pipeline.
    
-  Train encoder on 1–2k subset, export to ONNX.
    
-  Seed embeddings table and test nearest-neighbour retrieval.
    
-  Build frontend MediaPipe integration and test DTW word-mode.
    

---

If you want, I can now:

- Generate the **preprocessing Python script** (MediaPipe-based) to create the `.npy` keypoint files and CSV metadata, or
    
- Produce the **PyTorch training notebook** skeleton that implements the encoder and contrastive training loop.
    
