"""
build_dataset.py - Объединяет FMA (audio) + Song-Interpretation (lyrics) + Song Describer (descriptions)
"""

import os
import json
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from rapidfuzz import fuzz, process
import shutil
from collections import defaultdict
import random

# ========== НАСТРОЙКИ ==========
PROJECT_DIR = os.path.expanduser('~/music_fusion_project')
FMA_AUDIO_DIR = os.path.join(PROJECT_DIR, 'fma_large')
FMA_METADATA_PATH = os.path.join(PROJECT_DIR, 'fma_metadata', 'tracks.csv')
SONG_DESCRIBER_PATH = os.path.join(PROJECT_DIR, 'song_describer.csv')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'final_dataset')
AUDIO_OUTPUT = os.path.join(OUTPUT_DIR, 'audio')

os.makedirs(AUDIO_OUTPUT, exist_ok=True)

FUZZY_THRESHOLD = 80  # порог для матча по title

print("="*70)
print("BUILDING FUSION DATASET")
print("="*70)

# ========== 1. Song-Interpretation ==========
print("\n[1/6] Loading Song-Interpretation dataset (lyrics)...")
interp_ds = load_dataset("jamimulgrave/Song-Interpretation-Dataset", split="train")
enrich_ds = load_dataset("seungheondoh/enrich-music4all", split="train")
print(f"  ✓ Loaded {len(interp_ds)} tracks with lyrics")

enrich_map = {s['track_id']: s for s in enrich_ds}

lyrics_data = []
for sample in tqdm(interp_ds, desc="  Processing lyrics"):
    tid = sample['music4all_id']
    if tid in enrich_map:
        e = enrich_map[tid]
        artist = e['artist_name'].strip().lower()
        title  = e['title'].strip().lower()
        if not artist or not title:
            continue
        lyrics_data.append({
            'id': tid,
            'artist': artist,
            'title': title,
            'lyrics': sample['lyrics'],
            'user_comment': sample['comment'],
        })

print(f"  ✓ Prepared {len(lyrics_data)} tracks with lyrics")

# ========== 2. Song Describer ==========
print("\n[2/6] Loading Song Describer dataset (descriptions)...")
if os.path.exists(SONG_DESCRIBER_PATH):
    describer_df = pd.read_csv(SONG_DESCRIBER_PATH)
    print(f"  ✓ Loaded {len(describer_df)} tracks with descriptions")
    descriptions_data = []
    for _, row in describer_df.iterrows():
        artist = str(row.get('artist', '')).strip().lower()
        title  = str(row.get('title', '')).strip().lower()
        if not artist or not title:
            continue
        descriptions_data.append({
            'artist': artist,
            'title': title,
            'description': str(row.get('caption', '')),
        })
else:
    print("  ⚠️  Song Describer not found, using user comments as descriptions")
    descriptions_data = []

# ========== 3. FMA metadata ==========
print("\n[3/6] Loading FMA metadata (audio paths)...")
fma_tracks = pd.read_csv(FMA_METADATA_PATH, index_col=0, header=[0, 1])

fma_data = []
for track_id, row in tqdm(fma_tracks.iterrows(), total=len(fma_tracks), desc="  Processing FMA"):
    try:
        artist = str(row[('artist', 'name')]).strip().lower()
        title  = str(row[('track', 'title')]).strip().lower()
        if not artist or not title:
            continue
        track_id_str = f"{track_id:06d}"
        folder = track_id_str[:3]
        audio_path = os.path.join(FMA_AUDIO_DIR, folder, f"{track_id_str}.mp3")
        if os.path.exists(audio_path):
            fma_data.append({
                'fma_id': track_id,
                'artist': artist,
                'title': title,
                'audio_path': audio_path,
            })
    except Exception:
        continue

print(f"  ✓ FMA tracks with audio: {len(fma_data)}")

# индексы по артисту
fma_by_artist = defaultdict(list)
for item in fma_data:
    fma_by_artist[item['artist']].append(item)

descr_by_artist = defaultdict(list)
for item in descriptions_data:
    descr_by_artist[item['artist']].append(item)

# ========== функции матчинга ==========
def find_best_match_by_artist(query_artist, query_title, index_dict):
    """Поиск только среди треков этого артиста."""
    candidates = index_dict.get(query_artist, [])
    if not candidates:
        return None, 0
    best_match = None
    best_score = 0
    for c in candidates:
        score = fuzz.ratio(query_title, c['title'])
        if score > best_score:
            best_score = score
            best_match = c
    return best_match, best_score

def find_best_match_global(query_artist, query_title, candidates):
    """Fallback: ищем по всей выборке (artist+title одной строкой)."""
    if not candidates:
        return None, 0
    query = f"{query_artist} - {query_title}"
    choices = [f"{c['artist']} - {c['title']}" for c in candidates]
    best_str, best_score, best_idx = process.extractOne(
        query, choices, scorer=fuzz.ratio
    )
    return candidates[best_idx], best_score

# ========== 4. Matching ==========
print("\n[4/6] Matching tracks (artist + title, with fallback)...")
final_dataset = []

for lyrics_item in tqdm(lyrics_data, desc="  Matching"):
    artist = lyrics_item['artist']
    title  = lyrics_item['title']

    # 1) сначала ищем только среди такого же артиста
    fma_match, fma_score = find_best_match_by_artist(artist, title, fma_by_artist)

    # 2) если артиста нет или score ниже порога — fallback по всем трекам
    if (not fma_match) or (fma_score < FUZZY_THRESHOLD):
        fma_match, fma_score = find_best_match_global(artist, title, fma_data)

    # descriptions: только по артисту (fallback глобальный можно добавить по аналогии)
    if descriptions_data:
        desc_match, desc_score = find_best_match_by_artist(artist, title, descr_by_artist)
    else:
        desc_match, desc_score = (None, 0)

    if fma_match and fma_score >= FUZZY_THRESHOLD:
        description = (desc_match['description']
                       if (desc_match and desc_score >= FUZZY_THRESHOLD)
                       else lyrics_item['user_comment'])

        final_dataset.append({
            'track_id': lyrics_item['id'],
            'artist': lyrics_item['artist'],
            'title':  lyrics_item['title'],
            'lyrics': lyrics_item['lyrics'],
            'description': description,
            'audio_path': fma_match['audio_path'],
            'fma_id': fma_match['fma_id'],
            'match_scores': {
                'audio': fma_score,
                'description': desc_score if desc_match else None
            }
        })

print(f"  ✓ Matched {len(final_dataset)} tracks with all components")

# ========== 5. Копирование аудио ==========
print("\n[5/6] Copying audio files...")
for item in tqdm(final_dataset, desc="  Copying"):
    new_audio_path = os.path.join(AUDIO_OUTPUT, f"{item['track_id']}.mp3")
    try:
        shutil.copy2(item['audio_path'], new_audio_path)
        item['audio_path'] = new_audio_path
    except Exception as e:
        print(f"  Warning: Failed to copy {item['fma_id']}: {e}")

# ========== 6. Сохранение и сплиты ==========
print("\n[6/6] Saving dataset...")

full_file = os.path.join(OUTPUT_DIR, 'full_dataset.json')
with open(full_file, 'w', encoding='utf-8') as f:
    json.dump(final_dataset, f, ensure_ascii=False, indent=2)

random.seed(42)
random.shuffle(final_dataset)
n = len(final_dataset)
splits = {
    'train': final_dataset[:int(n*0.8)],
    'val':   final_dataset[int(n*0.8):int(n*0.9)],
    'test':  final_dataset[int(n*0.9):],
}
for name, split in splits.items():
    with open(os.path.join(OUTPUT_DIR, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=2)

print("\n" + "="*70)
print("✅ DATASET READY!")
print("="*70)
print(f"\n📊 Statistics:")
print(f"  Total tracks: {len(final_dataset)}")
print(f"  Train: {len(splits['train'])}")
print(f"  Val:   {len(splits['val'])}")
print(f"  Test:  {len(splits['test'])}")
print(f"\n📁 Saved to: {OUTPUT_DIR}")
