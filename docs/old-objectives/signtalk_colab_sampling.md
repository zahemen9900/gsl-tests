# 🎯 SignTalk-GH: Colab Sampling Notebook

**Dataset Structure Understanding:**
- Metadata.xlsx: 4,000 unique sentences (Sentence ID, Sentence, Text/Category)
- Videos folder: 10,000 videos (1A.mp4, 1B.mp4, 2A.mp4, etc.)
- Multiple video variations per sentence ID

---

## Cell 1 — Mount Drive & Install Dependencies

```python
from google.colab import drive
drive.mount('/content/drive')  # Optional: comment out if not using Drive

!pip install pandas openpyxl tqdm
import os, shutil, random, re
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
print("✅ Dependencies installed and ready")
```

---

## Cell 2 — Define Paths & Verify Dataset Structure

```python
# Adjust these paths to match your Drive structure
DATASET_DIR = "/content/drive/MyDrive/SignTalk-GH"  # Change if different
VIDEO_DIR = os.path.join(DATASET_DIR, "Videos")
META_PATH = os.path.join(DATASET_DIR, "Metadata.xlsx")

# Verify paths exist
assert os.path.exists(DATASET_DIR), f"❌ Dataset dir not found: {DATASET_DIR}"
assert os.path.exists(META_PATH), f"❌ Metadata.xlsx not found: {META_PATH}"
assert os.path.isdir(VIDEO_DIR), f"❌ Videos dir not found: {VIDEO_DIR}"

print(f"✅ Dataset directory: {DATASET_DIR}")
print(f"✅ Video directory: {VIDEO_DIR}")
print(f"✅ Metadata file: {META_PATH}")
```

---

## Cell 3 — Load Metadata (4,000 Unique Sentences)

```python
# Load the metadata Excel file
df_sentences = pd.read_excel(META_PATH, sheet_name="Sheet1")

print(f"📊 Loaded {len(df_sentences)} unique sentences")
print(f"📋 Columns: {df_sentences.columns.tolist()}")
print("\n📝 Sample rows:")
df_sentences.head()
```

---

## Cell 4 — Create Video-to-Sentence Mapping

**This is the CRITICAL step** - we scan all videos and map them to their sentence IDs.

```python
# Get all video files
video_files = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith('.mp4')]
print(f"📹 Found {len(video_files)} video files")

# Create video-to-sentence mapping
video_mapping = []
unmapped = []

# Pattern to extract sentence ID from filename (e.g., "1A.mp4" → 1)
pattern = re.compile(r'^(\d+)[A-Za-z]?\.mp4$', re.IGNORECASE)

def extract_variant_token(video_name: str, sentence_id: int) -> str:
    """Approximate signer/variant token from filename suffix."""
    stem = Path(video_name).stem
    suffix = stem.replace(str(sentence_id), "", 1).strip().upper()
    return suffix if suffix else "BASE"

for video_file in tqdm(video_files, desc="Mapping videos to sentences"):
    match = pattern.match(video_file)
    
    if not match:
        unmapped.append(video_file)
        continue
    
    # Extract sentence ID
    sentence_id = int(match.group(1))
    
    # Look up sentence info from metadata
    sentence_row = df_sentences[df_sentences['Sentence ID'] == sentence_id]
    
    if sentence_row.empty:
        unmapped.append(video_file)
        continue
    
    # Create mapping entry
    video_mapping.append({
        'video_file': video_file,
        'sentence_id': sentence_id,
        'sentence': sentence_row.iloc[0]['Sentence'],
        'category': sentence_row.iloc[0]['Text'],  # 'Text' column is the category
        'variant': extract_variant_token(video_file, sentence_id)
    })

# Create DataFrame from mapping
df_videos = pd.DataFrame(video_mapping)

print(f"\n✅ Successfully mapped {len(df_videos)} videos")
print(f"📊 Covering {df_videos['sentence_id'].nunique()} unique sentences")
print(f"📈 Average videos per sentence: {len(df_videos) / df_videos['sentence_id'].nunique():.2f}")
print(f"🎬 Approx variants per sentence: {df_videos.groupby('sentence_id')['variant'].nunique().mean():.2f}")
print(f"⚠️ Unmapped videos: {len(unmapped)}")

if unmapped:
    print(f"\n⚠️ Sample unmapped files: {unmapped[:5]}")
    pd.DataFrame({'unmapped_file': unmapped}).to_csv('/content/unmapped_videos.csv', index=False)

df_videos.head(10)
```

---

## Cell 5 — Stratified Sampling with Variant Coverage

Target balanced sampling while guaranteeing multiple variants per sentence when available.

```python
DOUBLE_SAMPLE = False  # Flip to True for ~2400 videos
TARGET_SAMPLE_COUNT = 2400 if DOUBLE_SAMPLE else 1200
MIN_VARIANTS_PER_SENTENCE = 2 if DOUBLE_SAMPLE else 1
MAX_VARIANTS_PER_SENTENCE = 4  # Avoid exploding duplicates

# Get category distribution
category_counts = df_videos['category'].value_counts()
print("📊 Category distribution in full dataset:")
print(category_counts)

print(f"\n🎯 Target clips: {TARGET_SAMPLE_COUNT}")
print(f"📋 Categories: {len(category_counts)}")
print(f"� Min variants per sentence: {MIN_VARIANTS_PER_SENTENCE}")

# Initial stratified sampling identical to baseline approach
num_categories = len(category_counts)
samples_per_category = TARGET_SAMPLE_COUNT // max(1, num_categories)
sampled_videos = []

for category in category_counts.index:
    category_videos = df_videos[df_videos['category'] == category]
    n_samples = min(len(category_videos), samples_per_category)
    if n_samples > 0:
        sampled = category_videos.sample(n=n_samples, random_state=RANDOM_SEED)
        sampled_videos.append(sampled)

df_sampled = pd.concat(sampled_videos, ignore_index=True)

# Top up if we are still short of the target
if len(df_sampled) < TARGET_SAMPLE_COUNT:
    remaining = df_videos[~df_videos['video_file'].isin(df_sampled['video_file'])]
    additional_needed = TARGET_SAMPLE_COUNT - len(df_sampled)
    if len(remaining) > 0:
        additional = remaining.sample(
            n=min(len(remaining), additional_needed),
            random_state=RANDOM_SEED
        )
        df_sampled = pd.concat([df_sampled, additional], ignore_index=True)

# Ensure sentences have the desired number of variants when available
if MIN_VARIANTS_PER_SENTENCE > 1:
    per_sentence_counts = df_sampled.groupby('sentence_id')['video_file'].count()
    needs_boost = per_sentence_counts[per_sentence_counts < MIN_VARIANTS_PER_SENTENCE].index.tolist()
    extras = []
    for sid in needs_boost:
        candidates = df_videos[
            (df_videos['sentence_id'] == sid) & (~df_videos['video_file'].isin(df_sampled['video_file']))
        ]
        if candidates.empty:
            continue
        unique_variants = candidates.groupby('variant', group_keys=False).head(1)
        take = min(
            MIN_VARIANTS_PER_SENTENCE - per_sentence_counts.get(sid, 0),
            len(unique_variants)
        )
        if take > 0:
            extras.append(unique_variants.sample(n=take, random_state=RANDOM_SEED))
    if extras:
        extra_df = pd.concat(extras, ignore_index=True)
        df_sampled = pd.concat([df_sampled, extra_df], ignore_index=True)

# Respect per-sentence cap
df_sampled = (
    df_sampled
    .sort_values('sentence_id')
    .drop_duplicates('video_file')
    .groupby('sentence_id', group_keys=False)
    .head(MAX_VARIANTS_PER_SENTENCE)
    .reset_index(drop=True)
)

# If we overshoot the target, drop whole sentences until we fit
if len(df_sampled) > TARGET_SAMPLE_COUNT:
    sentence_ids = df_sampled['sentence_id'].unique().tolist()
    random.Random(RANDOM_SEED).shuffle(sentence_ids)
    keep_ids = []
    running = 0
    for sid in sentence_ids:
        block = df_sampled[df_sampled['sentence_id'] == sid]
        block_len = len(block)
        if running + block_len <= TARGET_SAMPLE_COUNT:
            keep_ids.append(sid)
            running += block_len
    df_sampled = df_sampled[df_sampled['sentence_id'].isin(keep_ids)].reset_index(drop=True)

print(f"\n✅ Final sample size: {len(df_sampled)} videos")
print(f"📊 Covered sentences: {df_sampled['sentence_id'].nunique()}")
print("\n🎭 Variants per sentence (min/mean/max):",
      df_sampled.groupby('sentence_id')['video_file'].count().agg(['min', 'mean', 'max']).to_dict())
print("\n📋 Sampled category distribution:")
print(df_sampled['category'].value_counts())

df_sampled.head()
```

---

## Cell 6 — Copy Sampled Videos to Output Directory

```python
# Create output directory structure
OUTPUT_DIR = "/content/sample_dataset"
OUTPUT_VIDEO_DIR = os.path.join(OUTPUT_DIR, "videos")
os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)

# Copy files
copied_count = 0
failed_copies = []

for _, row in tqdm(df_sampled.iterrows(), total=len(df_sampled), desc="Copying videos"):
    src_path = os.path.join(VIDEO_DIR, row['video_file'])
    dst_path = os.path.join(OUTPUT_VIDEO_DIR, row['video_file'])
    
    try:
        shutil.copy2(src_path, dst_path)
        copied_count += 1
    except Exception as e:
        failed_copies.append({
            'video_file': row['video_file'],
            'error': str(e)
        })
        print(f"❌ Failed to copy {row['video_file']}: {e}")

print(f"\n✅ Successfully copied: {copied_count}/{len(df_sampled)} videos")
print(f"❌ Failed copies: {len(failed_copies)}")

if failed_copies:
    pd.DataFrame(failed_copies).to_csv(
        os.path.join(OUTPUT_DIR, "failed_copies.csv"), 
        index=False
    )
```

---

## Cell 7 — Save Metadata & Zip Dataset

```python
# Save sampled metadata
metadata_path = os.path.join(OUTPUT_DIR, "sampled_metadata.csv")
df_sampled.to_csv(metadata_path, index=False)
print(f"✅ Saved metadata: {metadata_path}")

# Create summary statistics
summary = {
    'total_videos': len(df_sampled),
    'unique_sentences': df_sampled['sentence_id'].nunique(),
    'categories': df_sampled['category'].nunique(),
    'videos_per_sentence_avg': len(df_sampled) / df_sampled['sentence_id'].nunique(),
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv(os.path.join(OUTPUT_DIR, "dataset_summary.csv"), index=False)
print("\n📊 Dataset Summary:")
print(summary_df.T)

# Zip the entire sampled dataset
!cd /content && zip -r SignTalk-GH_Sampled.zip sample_dataset/

print("\n✅ Zipped dataset: /content/SignTalk-GH_Sampled.zip")
print("📥 Download this file from Colab's file browser")
```

---

## Cell 8 — Verification & Stats

```python
# Final verification
print("=" * 60)
print("📊 SAMPLING SUMMARY")
print("=" * 60)

print(f"\n📹 Videos:")
print(f"  • Total sampled: {len(df_sampled)}")
print(f"  • Successfully copied: {copied_count}")
print(f"  • Failed: {len(failed_copies)}")

print(f"\n📝 Sentences:")
print(f"  • Unique sentences covered: {df_sampled['sentence_id'].nunique()}")
print(f"  • Original dataset sentences: {len(df_sentences)}")
print(f"  • Coverage: {df_sampled['sentence_id'].nunique() / len(df_sentences) * 100:.1f}%")

print(f"\n📋 Categories:")
print(df_sampled['category'].value_counts())

print(f"\n✅ Ready for local preprocessing!")
print(f"📦 Download: /content/SignTalk-GH_Sampled.zip")
```

---

## 🎯 What You Get

After running this notebook, you'll have:

1. **`SignTalk-GH_Sampled.zip`** containing:
   - `videos/` - ~1,200 sampled video files
   - `sampled_metadata.csv` - Video-to-sentence mapping with categories
   - `dataset_summary.csv` - Statistics about your sample

2. **Metadata structure:**
   - `video_file`: Actual video filename (e.g., "1A.mp4")
   - `sentence_id`: Numeric sentence ID (e.g., 1)
   - `sentence`: Text content of the sign
   - `category`: Healthcare category (e.g., "General Consultation")

3. **Ready for preprocessing** on your local machine with RTX 3050

---

## 📝 Notes

- Multiple videos (1A, 1B, 1C) mapping to the same sentence ID is EXPECTED
- This captures natural variation in signing
- Your model will learn from these variations
- Sampling is stratified by category for balanced training data
