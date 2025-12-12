# GSL Dictionary Dataset Integration

For your records, here is the exact logic that has been injected into your `testing-gsl-datasets.ipynb` notebook. This block is placed immediately after the metadata loading step and before video mapping.

It performs the following:
1.  **Parses the `GSL_openpose_data` directory**: It looks for folders like `DOCTOR`, `DOCTOR_2`, `RUNNY_NOSE`, etc.
2.  **Handles Synonyms**: `ABSTAIN_OR_AVOID` becomes "Abstain / Avoid".
3.  **Handles Variants**: `DOCTOR` and `DOCTOR_2` are merged under the same ID but with different video suffixes (e.g., `12345A.mp4` and `12345B.mp4`).
4.  **Renames & Moves**: Videos are standardized to the SignTalk schema (`{ID}{VARIANT}.mp4`) and copied to the main videos folder.
5.  **Fixes Metadata**: New rows are added to `df_sentences` so subsequent sampling steps include these new words.

## Code Block

```python
# --- GSL OPENPOSE DATASET INTEGRATION ---
print("\n🔄 Starting GSL Dictionary Dataset Integration...")

GSL_DIR = "/content/GSL_openpose_data"

if not os.path.exists(GSL_DIR):
    print(f"⚠️ GSL directory not found at {GSL_DIR}. Skipping integration.")
else:
    # 1. Parse GSL Folders
    # Structure: { "BASE_MEANING": [path_to_video1, path_to_video2, ...] }
    gsl_entries = {}
    
    # Regex to capture Base Name (Concept) and separate numeric suffix (e.g. _1, _2)
    # Ex: DOCTOR -> Group1: DOCTOR
    # Ex: DOCTOR_2 -> Group1: DOCTOR, Group2: _2
    # Ex: ABSTAIN_OR_AVOID -> Group1: ABSTAIN_OR_AVOID
    import re
    folder_pattern = re.compile(r'^(.+?)(_\d+)?$')

    all_folders = [d for d in os.listdir(GSL_DIR) if os.path.isdir(os.path.join(GSL_DIR, d))]
    
    for folder_name in tqdm(all_folders, desc="Parsing GSL Dictionary folders"):
        match = folder_pattern.match(folder_name)
        if not match:
            continue
            
        base_concept = match.group(1) # e.g. DOCTOR or ABSTAIN_OR_AVOID
        
        # Look for mp4 file inside
        folder_path = os.path.join(GSL_DIR, folder_name)
        mp4_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.mp4')]
        
        if not mp4_files:
            continue
            
        # We take the first mp4 found (assuming one main video per folder)
        video_path = os.path.join(folder_path, mp4_files[0])
        
        if base_concept not in gsl_entries:
            gsl_entries[base_concept] = []
        gsl_entries[base_concept].append(video_path)

    print(f"   Found {len(gsl_entries)} unique GSL dictionary concepts.")

    # 2. Process, Merge, and Rename
    current_id = df_sentences['Sentence ID'].max() + 1
    new_rows = []
    moved_count = 0
    
    # Helper to format text: COLD_OR_RUNNY_NOSE -> Cold / Runny Nose
    def format_concept_text(raw_text):
        parts = raw_text.split('_OR_')
        clean_parts = []
        for p in parts:
            # Replace underscores with spaces, unless it's a hyphenated phrase like FACE-TO-FACE
            # Heuristic: If it has a hyphen, don't replace underscores (likely none). 
            # If no hyphen, replace underscores with spaces (RUNNY_NOSE -> RUNNY NOSE)
            if '-' not in p:
                p = p.replace('_', ' ')
            clean_parts.append(p.lower().capitalize())
        return " / ".join(clean_parts)

    for base_concept, video_search_paths in tqdm(gsl_entries.items(), desc="Merging GSL Dictionary"):
        # Create Metadata Entry
        sentence_text = format_concept_text(base_concept)
        sid = current_id
        current_id += 1
        
        new_rows.append({
            'Sentence ID': sid,
            'Sentence Text': sentence_text,
            'Category': 'GSL Dictionary Word'
        })
        
        # Rename and Copy Variants (A, B, C...)
        variant_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        
        for idx, src_path in enumerate(video_search_paths):
            if idx >= len(variant_chars):
                break
            
            variant = variant_chars[idx]
            new_filename = f"{sid}{variant}.mp4"
            dst_path = os.path.join(VIDEO_DIR, new_filename)
            
            try:
                shutil.copy2(src_path, dst_path)
                moved_count += 1
            except Exception as e:
                print(f"❌ Error copying {src_path}: {e}")

    # 3. Update Metadata DataFrame
    if new_rows:
        df_new = pd.DataFrame(new_rows)
        df_sentences = pd.concat([df_sentences, df_new], ignore_index=True)
        print(f"\n✅ Integrated {len(new_rows)} new lexical items from GSL Dictionary.")
        print(f"   Copied {moved_count} videos to {VIDEO_DIR}.")
        print(f"   New Total Sentences: {len(df_sentences)}")
    else:
        print("ℹ️ No new items found to merge.")
```
