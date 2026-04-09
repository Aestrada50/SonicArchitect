# Sonic Architect — CLAUDE.md

## Project Overview
Multi-class music genre classifier that identifies a song's genre from its audio fingerprint (tempo, energy, danceability, etc.). Think of it as translating "vibes" into math.

## Project Structure
```
SonicArchitect/
├── data/
│   ├── GenreFeatures.csv       # 114k tracks, primary training data
│   └── SpotifySongs.csv       # GTZAN benchmark dataset (features_30_sec.csv)
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory data analysis
│   ├── 02_preprocessing.ipynb   # Cleaning, feature selection, encoding
│   ├── 03_training.ipynb        # Model training & evaluation
│   └── 04_visualization.ipynb   # Vibe Correlation Map & results
├── models/
│   └── random_forest.pkl        # Saved trained model
├── venv/                        # Python virtual environment (do not modify)
└── CLAUDE.md
```

## Stack
- **Language**: Python 3.14 (Homebrew)
- **Environment**: venv — always activate before running anything
- **Editor**: VS Code with Jupyter (.ipynb) extension
- **Key Libraries**: pandas, numpy, scikit-learn, matplotlib, seaborn

## Activate Environment
```bash
source venv/bin/activate
```

## Datasets
- **Spotify Songs Dataset** (maharshipandya on Kaggle): 114,000 tracks, pre-extracted audio features, clean tabular format. Primary training data.
- **Genre Features Dataset** (andradaolteanu on Kaggle): Standard ML benchmark, 10 genres

## Key Audio Features
These are the input features the model trains on:
- `danceability` — how suitable for dancing (0.0–1.0)
- `energy` — intensity and activity (0.0–1.0)
- `acousticness` — whether the track is acoustic (0.0–1.0)
- `valence` — musical positiveness (0.0–1.0)
- `speechiness` — presence of spoken words (0.0–1.0)
- `tempo` — BPM
- `loudness` — overall loudness in dB
- `instrumentalness` — likelihood of no vocals (0.0–1.0)
- `liveness` — presence of live audience (0.0–1.0)

## ML Approach
1. **Baseline model**: Random Forest classifier
2. **Target metric**: F1-Score (macro average) across at least 5 genres
3. **Why F1**: Genre classes are imbalanced — accuracy alone is misleading
4. **Train/test split**: 80/20 stratified by genre

## Goals
- [ ] Trained Random Forest with strong F1-Score across 5+ genres
- [ ] Vibe Correlation Map — radar/spider chart showing audio DNA per genre
- [ ] Real-world validation — feed the model an unseen song and check the prediction
- [ ] Stretch goal: Spotipy integration for real-time classification via Spotify Web API

## Coding Conventions
- One notebook per phase (EDA → Preprocessing → Training → Visualization)
- Save the trained model with `joblib` to `models/random_forest.pkl`
- Keep all file paths relative to the project root
- Comment every major code block — this is a class project

## Notes for Claude Code
- Do not modify anything inside `venv/`
- All data lives in `data/` — never hardcode absolute paths
- When generating code, default to scikit-learn unless told otherwise
- Prefer clear readable code over clever one-liners — this is for a grade