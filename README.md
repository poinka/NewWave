# D1.1 Progress Submission

## Team Information

- Project name: New Wave

### Participants

- Janna Ivanova — j.ivanova@innopolis.university
- Polina Korobeinikova — p.korobeinikova@innopolis.university
- Vladislav Kalinichenko — v.kalinichenko@innopolis.university

## Project Topic & Description

Our project focuses on building a music-by-description retrieval system that allows users to find modern songs by providing free-form natural language queries. Our initial approach is to combine CLAP for audio–text alignment, a separate lyrics encoder for semantic understanding of song lyrics, and Gemma-3 270M as a lightweight query rewriter to transform vague user inputs into structured musical descriptors (mood, genre, instrumentation, vocal style).

The final goal is to let users search music collections using intuitive descriptions such as “a melancholic indie ballad with soft vocals and slow tempo” and retrieve songs that match both their sound and lyrical content. This is important because existing search interfaces (by title, artist, or fixed tags) fail to capture the richness of human intent, especially when users only know the “feeling” or themes they want.

### Target Users

- Music listeners seeking personalized discovery beyond metadata or charts.
- Content creators (e.g., video editors, game developers) needing to quickly find tracks by mood/energy.
- Music platforms aiming to improve recommendation quality and user engagement.

### Value Proposition

The project is valuable because it explores cross-modal learning for long-form audio, integrates semantic lyric matching, and demonstrates how modern generative models like Gemma can bridge vague human input with machine-understandable embeddings.

## Dataset

- MusicCaps (Google / Hugging Face)
- The Song Describer Dataset (SDD)
- Jamendo CC catalog (optional)

## Success Criteria & Metrics

We will measure retrieval effectiveness using standard information retrieval metrics. Target thresholds reflect our project goals and available resources.

### Intrinsic Retrieval Metrics

- Recall@K (K = 5, 10, 20): proportion of relevant songs in top K > 70%
- nDCG@K: normalized discounted cumulative gain > 70%
- MRR (Mean Reciprocal Rank): average reciprocal rank of the first relevant song > 70%
- Precision@K: proportion of retrieved songs in top K that are relevant > 70%

### Proxy Labels for Ground Truth

- Use tags, genres, and moods from external APIs such as Spotify audio features, Last.fm tags, and MusicBrainz tags as reference labels
- Matching process: query → tag-based description → compare with track tags
- Relevance is measured as either tag overlap or cosine similarity between the embedding of the description and the embedding of the track tags

Important Note: We do not expect retrieval performance to cover queries that explicitly mention song metadata (e.g., artist names, track titles, release years). Our system operates solely on content-based matching, driven by audio and lyrical semantics rather than metadata lookup.

## Work Distribution

- Janna Ivanova – research on related works and competitors, explores technologies, writes documentation and reports.
- Polina Korobeinikova – implementation, experiments with frameworks and tools, coding the application together with Vladislav.
- Vladislav Kalinichenko – theoretical and mathematical analysis of methods, evaluates model designs, contributes to coding.

## Competitor & SOTA Research

1) Goal & Positioning

The project tackles music search via free-form natural-language descriptions. Architecture: audio–text alignment (CLAP), a dedicated lyrics encoder (semantic understanding of song texts), and a lightweight query rewriter (Gemma‑3 270M) that normalizes user input into attributes: mood, genre, instrumentation, vocal style/timbre. Core value: strict relevance both in sound and in lyrical meaning, honoring even small, non‑obvious user preferences.

2) SOTA Landscape & Technology Trends

The current wave combines audio↔text joint embeddings and generative retrieval. MuLan demonstrated large‑scale zero‑shot alignment between music and free text across tens of millions of tracks. CLAP brought contrastive sound↔text pretraining to the open stack, but was trained largely on short audio clips (~10 s), which limits long‑form song understanding without adaptation. Generative retrieval (Text2Tracks): an LLM directly generates track IDs from a textual prompt, showing substantial gains in internal benchmarks. Multimodal personalizers (e.g., JAM) unify audio, lyrics, and collaborative signals into a single latent space for fine‑grained matching. Why lyrics matter: Spotify’s research indicates lyrics often correlate with song “mood” as much as (and sometimes more than) audio alone; the best results come from combining both (Spotify ICWSM’22 blog, paper PDF).

Note on CLAP and our plan

We plan to fine‑tune/adapt CLAP on full‑length music tracks and add temporal aggregation improvements so the model captures song‑level form, dynamics, and vocal nuances—not just short clips (e.g., AudioSet 10‑second clips).

3) “My Wave” (Yandex Music): technology & relevance

“My Wave” is an infinite personalized feed that adapts to likes/skips, audio features, and behavioral signals. Tech profile: a hybrid of collaborative and content models focused on personalized streaming (see AI DJ‑mix sets), not on explicit description‑based search.

Goal alignment: “My Wave” does not accept arbitrary descriptive queries. Our system centers on explicit natural‑language input, decomposes it into attributes, and validates matches via both audio and lyrics.

4) Closest Competitors (brief profiles)

- Spotify: semantic search and research on prompt‑to‑track‑IDs (Text2Tracks) trained on playlists. Strength: massive data. Limitation: limited transparency of attribute control.
- Cyanite.ai: auto‑tagging (genre, mood, instruments, lyric theme) with free‑text search over catalogs. Limitation: gaps on complex prompts and fine lyrical/vocal nuances.
- Musiio: large‑scale auto‑tagging. Limitation: no full semantic lyrics encoder for free‑text → joint audio+lyrics matching.
- Pandora Genome: human‑curated attributes provide strong similarity but do not scale and are closed.
- Research models: MuLan, CLAP, JAM provide foundations for zero‑shot music‑by‑description and multimodal personalization; our configuration follows this line while adding an LLM‑based query normalizer.

5) Comparison Table (summary)

System | User input | Lyrics use | Attribute control | Long songs | Primary focus
--- | --- | --- | --- | --- | ---
Our project | Free text → Gemma descriptors | Yes | High (explicit per‑attribute match) | Yes (CLAP fine‑tuning + segment aggregation) | Exact match to description
My Wave (Yandex) | Implicit: behavior/clicks | Partial/implicit | Medium | Yes | Personalized stream
Spotify | Free text (research) | Indirect | Medium | Yes | Prompt‑to‑tracks
Cyanite.ai | Free text & references | Partial (lyric theme) | Medium/High | Yes | B2B catalogs
Musiio | Tags | No full semantic encoder | Medium | Yes | Large‑scale filtering
Pandora | Reference track | Partial | High, but closed | Yes | Attribute similarity

6) Our Advantage in Practice

We merge the strongest ideas: LLM‑based query parsing → joint audio+lyrics matching → CLAP adapted to long songs. The result is retrieval that faithfully executes the user’s intent, including rare and non‑trivial wishes (tempo, instrumentation, vocal character, genre niche, thematic/affective cues in lyrics), rather than “roughly similar” results.

## Next Steps

1. Prepare CLAP adaptation plan for long-form songs: data collection, segment-level aggregation strategy, and training schedule.
2. Implement a lyrics encoder prototype and test semantic alignment with CLAP embeddings.
3. Integrate Gemma-3 270M for query rewriting; collect examples and fine-tune prompt templates.
4. Build a small retrieval demo with MusicCaps and SDD to validate recall@K and nDCG metrics.

## How to run the API locally (quick start)

0. Get your Mac IP via a `ipconfig getifaddr en0` command and paste it to `New-Wave-Info.plist` file
1. `cd "/Users/vladislavkalinichenko/VSCodeProjects/new wave/NewWave"`
2. `source .venv/bin/activate && uvicorn api:app --host 0.0.0.0 --port 8888`
3. (new terminal) `curl http://127.0.0.1:8888/health`

For a phone on the same Wi‑Fi: `ipconfig getifaddr en0` → use `http://<that-ip>:8888`.

## iOS client configuration

1. In Xcode, open the **New Wave** target → **Info** tab → add a `String` key named `VectorAPIBaseURL`.
2. Set its value to the Mac’s LAN URL, e.g. `http://10.91.49.14:8888`.
3. Rebuild/install the app. The client reads this value at launch, so change it whenever the server IP changes.
