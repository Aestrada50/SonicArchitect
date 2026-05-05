"""
umap_diagnostic.py
------------------
Diagnostic visualizer for SonicArchitect.

Runs UMAP on all ~114k Spotify tracks (15 audio features, scaled),
then renders an interactive Plotly figure where:
  - A dropdown recolors points by any of the 15 features (Viridis scale)
  - The genre legend allows toggling individual genres on/off
  - Hover shows track_name, artists, and the selected feature value

Usage:
    source venv/bin/activate
    python Notebooks/umap_diagnostic.py
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from umap import UMAP

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Path to dataset — relative to project root, script is in Notebooks/
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "Data", "SpotifySongs.csv")

# All 12 audio features the model trains on
FEATURES = [
    'popularity',
    'duration_ms',
    'mode',
    'danceability',
    'energy',
    'instrumentalness',
    'liveness',
    'valence',
    'tempo',
    'loudness',
    'speechiness',
    'acousticness',
]

# Metadata columns kept alongside features
META_COLS = ["track_genre", "track_name", "artists"]

# UMAP settings
UMAP_NEIGHBORS = 15
UMAP_MIN_DIST  = 0.0
UMAP_COMPONENTS = 2
UMAP_SEED = 42

# The 30 genres shown in both figures — covers rock, electronic, urban, acoustic, and pop families
GENRES_30 = [     'acoustic',   'alternative',   'black-metal',          'jazz',
    'electronic', 'chicago-house',     'classical',     'dancehall',
       'hip-hop',           'pop',        'techno',        'trance']

# Visualization settings
SAMPLES_PER_GENRE = 200   # max tracks per genre in the plot
MARKER_SIZE = 2
MARKER_OPACITY = 0.6
DEFAULT_FEATURE = "danceability"   # feature shown on first load
COLORSCALE = "Viridis"

# ---------------------------------------------------------------------------
# Step 1 — Load and scale data
# ---------------------------------------------------------------------------

def load_and_scale(data_path: str) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load SpotifySongs.csv, drop NaN rows in FEATURES, and return:
      - df:      full cleaned DataFrame (features + metadata columns)
      - X_scaled: numpy array of shape (n_rows, 15), StandardScaler applied

    We refit the scaler on the FULL dataset (not just the training split)
    so the UMAP projection covers the full data distribution.
    """
    print("Loading data...")
    df = pd.read_csv(data_path, usecols=FEATURES + META_COLS)

    # Drop any row missing a feature value
    before = len(df)
    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    print(f"  Loaded {len(df):,} rows ({before - len(df):,} dropped for NaN)")

    # Scale features — fit on the full dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES].values)
    print(f"  Features scaled: {X_scaled.shape}")

    return df, X_scaled


# ---------------------------------------------------------------------------
# Step 2 — Fit UMAP on full dataset
# ---------------------------------------------------------------------------

def fit_umap(X_scaled: np.ndarray) -> np.ndarray:
    """
    Fit UMAP on all rows of X_scaled (no PCA pre-processing).
    Returns embedding of shape (n_rows, 2).

    min_dist=0.0 allows points in the same cluster to collide,
    making dense regions of genre space easier to see.
    """
    print(f"Fitting UMAP on {X_scaled.shape[0]:,} rows × {X_scaled.shape[1]} features...")
    reducer = UMAP(
        n_neighbors=UMAP_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=UMAP_COMPONENTS,
        random_state=UMAP_SEED,
        # Spectral init fails on this dataset due to discrete/low-variance features
        # (explicit, key, mode, time_signature) causing a degenerate eigengap.
        # Random init is equivalent here since random_state ensures reproducibility.
        init="random",
    )
    embedding = reducer.fit_transform(X_scaled)
    print(f"  UMAP done. Embedding shape: {embedding.shape}")
    return embedding


# ---------------------------------------------------------------------------
# Step 3 — Stratified sample for visualization
# ---------------------------------------------------------------------------

def stratified_sample(
    df: pd.DataFrame,
    embedding: np.ndarray,
    samples_per_genre: int,
) -> pd.DataFrame:
    """
    Sample up to `samples_per_genre` rows per genre.
    Attaches the 2D UMAP coordinates as columns 'umap_x' and 'umap_y'.
    Returns a new DataFrame with only the sampled rows.
    """
    # Attach UMAP coords to the DataFrame before sampling
    df = df.copy()
    df["umap_x"] = embedding[:, 0]
    df["umap_y"] = embedding[:, 1]

    # pandas 2.x changed groupby().apply() to exclude the key column from each group,
    # which drops 'track_genre' from the result. Using a concat loop instead — each
    # group 'g' from groupby iteration retains all columns including 'track_genre'.
    sampled = pd.concat(
        [g.sample(min(len(g), samples_per_genre), random_state=UMAP_SEED)
         for _, g in df.groupby("track_genre")],
        ignore_index=True,
    )
    print(f"  Sampled {len(sampled):,} points across {sampled['track_genre'].nunique()} genres")
    return sampled


# ---------------------------------------------------------------------------
# Step 4 — Build Plotly figure
# ---------------------------------------------------------------------------

def build_figure(sampled: pd.DataFrame) -> go.Figure:
    """
    Build an interactive Plotly figure with:
      - One Scattergl trace per genre (enables native legend toggling)
      - Dropdown to recolor all traces by any of the 15 features
      - Shared Viridis colorscale (cmin/cmax consistent across traces)
      - Hover: track_name, artists, selected feature value
    """
    genres = sorted(sampled["track_genre"].unique())
    fig = go.Figure()

    # --- Compute global color range for the default feature ---
    # We use the sampled data range so the colorscale reflects what's visible.
    global_cmin = sampled[DEFAULT_FEATURE].min()
    global_cmax = sampled[DEFAULT_FEATURE].max()

    # --- One Scattergl trace per genre ---
    for genre in genres:
        mask = sampled["track_genre"] == genre
        g = sampled[mask]

        fig.add_trace(go.Scattergl(
            x=g["umap_x"],
            y=g["umap_y"],
            mode="markers",
            name=genre,          # shown in the legend
            legendgroup=genre,
            marker=dict(
                size=MARKER_SIZE,
                opacity=MARKER_OPACITY,
                color=g[DEFAULT_FEATURE].values,   # initial coloring
                colorscale=COLORSCALE,
                cmin=global_cmin,
                cmax=global_cmax,
                showscale=False,   # colorbar comes from the dummy trace below
            ),
            # Hover: track_name | artists | feature value
            customdata=np.stack([
                g["track_name"].values,
                g["artists"].values,
                g[DEFAULT_FEATURE].values,
            ], axis=1),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Artist: %{customdata[1]}<br>"
                f"{DEFAULT_FEATURE}: %{{customdata[2]}}<br>"
                "<extra></extra>"
            ),
            text=g["track_name"],
        ))

    # --- Invisible dummy trace that owns the shared colorbar ---
    # This is a single-point invisible trace whose only job is to render
    # the Viridis colorbar on the right side of the figure.
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(
            colorscale=COLORSCALE,
            cmin=global_cmin,
            cmax=global_cmax,
            showscale=True,
            colorbar=dict(title=DEFAULT_FEATURE, thickness=15, len=0.7),
            color=[global_cmin],   # single point, won't be visible
        ),
        showlegend=False,
        hoverinfo="skip",
        name="colorbar_dummy",
    ))

    # --- Dropdown buttons (one per feature) ---
    n_genre_traces = len(genres)   # the dummy trace is at index n_genre_traces
    buttons = []

    for feat in FEATURES:
        feat_min = sampled[feat].min()
        feat_max = sampled[feat].max()

        # marker.color for each genre trace — list of arrays, one per trace
        colors_per_trace = []
        customdata_per_trace = []
        hovertemplate_per_trace = []

        for genre in genres:
            mask = sampled["track_genre"] == genre
            g = sampled[mask]
            colors_per_trace.append(g[feat].values)

            customdata_per_trace.append(np.stack([
                g["track_name"].values,
                g["artists"].values,
                g[feat].values,
            ], axis=1))

            hovertemplate_per_trace.append(
                "<b>%{customdata[0]}</b><br>"
                f"Artist: %{{customdata[1]}}<br>"
                f"{feat}: %{{customdata[2]}}<br>"
                "<extra></extra>"
            )

        # restyle args: update all genre traces + the dummy colorbar trace
        buttons.append(dict(
            label=feat,
            method="restyle",
            args=[
                {
                    # Update genre traces (indices 0..n_genre_traces-1)
                    # Use [feat_min] (not None) so the dummy colorbar trace keeps a valid color value
                    "marker.color": colors_per_trace + [[feat_min]],
                    "marker.cmin":  [feat_min] * n_genre_traces + [feat_min],
                    "marker.cmax":  [feat_max] * n_genre_traces + [feat_max],
                    "customdata":   customdata_per_trace + [None],
                    "hovertemplate": hovertemplate_per_trace + [None],
                    # Update colorbar label on the dummy trace
                    "marker.colorbar.title.text": [None] * n_genre_traces + [feat],
                },
            ],
        ))

    fig.update_layout(
        title="SonicArchitect — UMAP Feature Diagnostic",
        updatemenus=[dict(
            buttons=buttons,
            direction="down",
            showactive=True,
            x=0.01,
            xanchor="left",
            y=1.12,
            yanchor="top",
            bgcolor="white",
            bordercolor="#ccc",
        )],
        annotations=[dict(
            text="Color by feature:",
            x=0.01, xanchor="left",
            y=1.16, yanchor="top",
            xref="paper", yref="paper",
            showarrow=False,
        )],
        legend=dict(
            title="Genre",
            itemsizing="constant",
            tracegroupgap=2,
        ),
        hovermode="closest",
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#16213e",
        font_color="black",
        margin=dict(t=100, r=120),
        xaxis=dict(title="UMAP-1", showgrid=False, zeroline=False),
        yaxis=dict(title="UMAP-2", showgrid=False, zeroline=False),
    )

    return fig


# ---------------------------------------------------------------------------
# Step 5 — Build genre-colored scatter plot
# ---------------------------------------------------------------------------

def build_genre_figure(sampled: pd.DataFrame) -> go.Figure:
    """
    Build a second Plotly figure where each point is colored by its genre label.

    Design choices:
    - One Scattergl trace per genre so the legend is natively toggle-able
      (clicking a genre name hides/shows that cluster)
    - Markers are small (size=2) and semi-transparent (opacity=0.4)
      to handle 22k overlapping points without a wall of ink
    - Colors cycle through Plotly's 26-color Alphabet palette; with 114 genres
      some colors repeat, but nearby genres in the sorted list are still
      visually distinct enough to see cluster boundaries
    - The legend is scrollable — no extra UI needed for 114 genres
    """
    import plotly.colors as pc

    # Alphabet has 26 distinct colors; we cycle it for all 114 genres
    palette = pc.qualitative.Alphabet
    genres  = sorted(sampled["track_genre"].unique())

    fig = go.Figure()

    for i, genre in enumerate(genres):
        mask  = sampled["track_genre"] == genre
        g     = sampled[mask]
        color = palette[i % len(palette)]   # cycle through the 26-color palette

        fig.add_trace(go.Scattergl(
            x=g["umap_x"],
            y=g["umap_y"],
            mode="markers",
            name=genre,          # shows up in the legend
            legendgroup=genre,
            marker=dict(
                size=2,
                opacity=0.4,
                color=color,
            ),
            # Hover shows track name and genre — enough to orient without clutter
            customdata=np.stack([
                g["track_name"].values,
                g["track_genre"].values,
            ], axis=1),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Genre: %{customdata[1]}<br>"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="UMAP — Genre Clusters",
        legend=dict(
            title="Genre (click to toggle)",
            itemsizing="constant",   # keeps legend dots the same size regardless of marker size
            tracegroupgap=1,
        ),
        hovermode="closest",
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#16213e",
        font_color="white",
        margin=dict(t=80, r=160),   # extra right margin so the legend doesn't overlap points
        xaxis=dict(title="UMAP-1", showgrid=False, zeroline=False),
        yaxis=dict(title="UMAP-2", showgrid=False, zeroline=False),
    )

    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # 1. Load and scale
    df, X_scaled = load_and_scale(DATA_PATH)

    # 2. Filter to the 30 target genres before UMAP runs
    valid_genres = [g for g in GENRES_30 if g in df["track_genre"].unique()]
    missing      = set(GENRES_30) - set(valid_genres)
    if missing:
        print(f"  Warning: {len(missing)} genres not found in data and will be skipped: {sorted(missing)}")
    mask     = df["track_genre"].isin(valid_genres)
    df       = df[mask].reset_index(drop=True)
    X_scaled = X_scaled[mask]
    print(f"  Filtered to {len(df):,} rows across {df['track_genre'].nunique()} genres")

    # 3. UMAP fit on the 30-genre subset only
    embedding = fit_umap(X_scaled)

    # 4. Stratified sample for visualization
    sampled = stratified_sample(df, embedding, SAMPLES_PER_GENRE)

    # 5. Build and show the feature-coloring diagnostic figure
    print("Building feature diagnostic figure...")
    fig_features = build_figure(sampled)
    print("Opening feature diagnostic in browser...")
    fig_features.show()

    # 6. Build and show the genre-cluster figure (separate browser tab/window)
    print("Building genre cluster figure...")
    fig_genres = build_genre_figure(sampled)
    print("Opening genre cluster figure in browser...")
    fig_genres.show()


if __name__ == "__main__":
    main()
