# Music Dataset Visualization & Cleaning
# Datasets: enrich-music4all & Song-Interpretation-Dataset

# ============================================================================
# SETUP
# ============================================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# LOAD DATASETS
# ============================================================================
# Load datasets
interpretation_ds = load_dataset("jamimulgrave/Song-Interpretation-Dataset")['train']
enrich_ds = load_dataset("seungheondoh/enrich-music4all")['train']

print(f"Interpretation dataset size: {len(interpretation_ds)}")
print(f"Enrich dataset size: {len(enrich_ds)}")

# ============================================================================
# CREATE MAPPINGS
# ============================================================================
# Create mappings from enrich dataset
pseudo_map = {row['track_id']: row['pseudo_caption'] for row in enrich_ds}
artist_map = {row['track_id']: row['artist_name'] for row in enrich_ds}
tag_map = {row['track_id']: row.get('tag_list', []) for row in enrich_ds}

print(f"Mappings created: {len(pseudo_map)} tracks")

# ============================================================================
# EXTRACT AND STRUCTURE DATA
# ============================================================================
# Extract key fields
music4all_ids = interpretation_ds['music4all_id']
descriptions = interpretation_ds['comment']
lyrics_list = interpretation_ds['lyrics']

# Create combined dataframe
data = []
for idx, track_id in enumerate(music4all_ids):
    data.append({
        'track_id': track_id,
        'user_interpretation': descriptions[idx],
        'lyrics': lyrics_list[idx],
        'pseudo_caption': pseudo_map.get(track_id, ''),
        'artist': artist_map.get(track_id, ''),
        'tags': tag_map.get(track_id, [])
    })

df = pd.DataFrame(data)
print(f"Combined dataset shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")

# ============================================================================
# DATA QUALITY CHECKS
# ============================================================================
# Check for missing data
print("\n" + "="*80)
print("DATA OVERVIEW:")
print("="*80)
print(f"Total rows: {len(df)}")
print(f"\nRows with empty pseudo_caption: {(df['pseudo_caption'] == '').sum()}")
print(f"Rows with empty user_interpretation: {(df['user_interpretation'] == '').sum()}")
print(f"Rows with empty lyrics: {(df['lyrics'] == '').sum()}")

# Text length statistics
df['interpretation_length'] = df['user_interpretation'].str.len()
df['lyrics_length'] = df['lyrics'].str.len()
df['caption_length'] = df['pseudo_caption'].str.len()

print("\n" + "="*80)
print("TEXT LENGTH STATISTICS:")
print("="*80)
print(df[['interpretation_length', 'lyrics_length', 'caption_length']].describe())

# ============================================================================
# VISUALIZE DATA DISTRIBUTION
# ============================================================================
# Text length distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(df['interpretation_length'], bins=50, edgecolor='black', alpha=0.7)
axes[0].set_title('User Interpretation Length', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Characters')
axes[0].set_ylabel('Count')

axes[1].hist(df['lyrics_length'], bins=50, edgecolor='black', alpha=0.7, color='orange')
axes[1].set_title('Lyrics Length', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Characters')

axes[2].hist(df['caption_length'], bins=50, edgecolor='black', alpha=0.7, color='green')
axes[2].set_title('Pseudo Caption Length', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Characters')

plt.tight_layout()
plt.show()

# ============================================================================
# TAG ANALYSIS
# ============================================================================
# Extract and count all tags
all_tags = []
for tags in df['tags']:
    if isinstance(tags, list):
        all_tags.extend(tags)

tag_counts = Counter(all_tags)
print("\n" + "="*80)
print("TAG STATISTICS:")
print("="*80)
print(f"Total unique tags: {len(tag_counts)}")
print(f"Total tag occurrences: {len(all_tags)}")
print(f"\nTop 20 tags:")
for tag, count in tag_counts.most_common(20):
    print(f"  {tag:30} {count:5}")

# ============================================================================
# VISUALIZE TOP TAGS
# ============================================================================
# Plot top 30 tags
top_tags = tag_counts.most_common(30)
tags, counts = zip(*top_tags)

plt.figure(figsize=(12, 8))
plt.barh(range(len(tags)), counts, color='steelblue', edgecolor='black')
plt.yticks(range(len(tags)), tags, fontsize=10)
plt.xlabel('Count', fontsize=12)
plt.title('Top 30 Music Tags', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ============================================================================
# CLEAN DATASET
# ============================================================================
print("\n" + "="*80)
print("CLEANING DATASET:")
print("="*80)

# Remove rows only if missing lyrics OR missing both descriptions
df_clean = df[
    (df['lyrics'] != '') & 
    ((df['pseudo_caption'] != '') | (df['user_interpretation'] != ''))
].copy()

print(f"Original size: {len(df)}")
print(f"Cleaned size: {len(df_clean)}")


# ============================================================================
# SPLIT DATASET
# ============================================================================
print("\n" + "="*80)
print("DATASET SPLIT:")
print("="*80)

# Train/Val/Test split (80/10/10)
num_samples = len(df_clean)
train_idx = int(0.8 * num_samples)
val_idx = int(0.9 * num_samples)

train_df = df_clean[:train_idx].reset_index(drop=True)
val_df = df_clean[train_idx:val_idx].reset_index(drop=True)
test_df = df_clean[val_idx:].reset_index(drop=True)

print(f"Train: {len(train_df):5} ({100*len(train_df)/num_samples:.1f}%)")
print(f"Val:   {len(val_df):5} ({100*len(val_df)/num_samples:.1f}%)")
print(f"Test:  {len(test_df):5} ({100*len(test_df)/num_samples:.1f}%)")

# ============================================================================
# SAMPLE DATA INSPECTION
# ============================================================================
print("\n" + "="*80)
print("SAMPLE DATA:")
print("="*80)

# Show random samples
sample = df_clean.sample(min(3, len(df_clean)))
for idx, row in sample.iterrows():
    print("\n" + "-"*80)
    print(f"Track ID: {row['track_id']}")
    print(f"Artist: {row['artist']}")
    print(f"Tags: {', '.join(row['tags'][:5]) if row['tags'] else 'None'}")
    print(f"\nPseudo Caption ({len(row['pseudo_caption'])} chars):")
    print(f"  {row['pseudo_caption'][:200] if row['pseudo_caption'] else '[EMPTY]'}...")
    print(f"\nUser Interpretation ({len(row['user_interpretation'])} chars):")
    print(f"  {row['user_interpretation'][:200] if row['user_interpretation'] else '[EMPTY]'}...")
    print(f"\nLyrics ({len(row['lyrics'])} chars):")
    print(f"  {row['lyrics'][:200]}...")

# ============================================================================
# EXPORT CLEAN DATA
# ============================================================================
print("\n" + "="*80)
print("EXPORTING DATA:")
print("="*80)

# Save cleaned datasets
train_df.to_csv('train_clean.csv', index=False)
val_df.to_csv('val_clean.csv', index=False)
test_df.to_csv('test_clean.csv', index=False)

print("✓ Saved cleaned datasets:")
print("  - train_clean.csv")
print("  - val_clean.csv")
print("  - test_clean.csv")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*80)
print("FINAL DATASET SUMMARY:")
print("="*80)
print(f"Total tracks: {len(df_clean)}")
print(f"Unique artists: {df_clean['artist'].nunique()}")
print(f"\nAverage lengths:")
print(f"  - User interpretation: {df_clean['interpretation_length'].mean():.0f} chars")
print(f"  - Lyrics: {df_clean['lyrics_length'].mean():.0f} chars")
print(f"  - Pseudo caption: {df_clean['caption_length'].mean():.0f} chars")

# Check description availability
has_both = ((df_clean['pseudo_caption'] != '') & (df_clean['user_interpretation'] != '')).sum()
has_pseudo_only = ((df_clean['pseudo_caption'] != '') & (df_clean['user_interpretation'] == '')).sum()
has_user_only = ((df_clean['pseudo_caption'] == '') & (df_clean['user_interpretation'] != '')).sum()

print(f"\nDescription availability:")
print(f"  - Has both descriptions: {has_both} ({100*has_both/len(df_clean):.1f}%)")
print(f"  - Has pseudo_caption only: {has_pseudo_only} ({100*has_pseudo_only/len(df_clean):.1f}%)")
print(f"  - Has user_interpretation only: {has_user_only} ({100*has_user_only/len(df_clean):.1f}%)")

# Distribution by split
print(f"\nTag statistics per split:")
print(f"  - Train: {sum(len(tags) for tags in train_df['tags'] if tags)} total tags")
print(f"  - Val:   {sum(len(tags) for tags in val_df['tags'] if tags)} total tags")
print(f"  - Test:  {sum(len(tags) for tags in test_df['tags'] if tags)} total tags")

print("\n" + "="*80)
print("✓ PROCESSING COMPLETE!")
print("="*80)
