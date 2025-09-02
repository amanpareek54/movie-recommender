import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- Load dataset safely --------------------
movies_df = pd.read_csv("movies.csv", dtype=str, keep_default_na=False)

# -------------------- Helper functions --------------------
def nz(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s

def split_multi(s: str) -> list[str]:
    s = nz(s)
    if not s:
        return []
    s = re.sub(r"[;,/]+", "|", s)
    s = re.sub(r"\|{2,}", "|", s)
    return [p.strip() for p in s.split("|") if p.strip()]

def top2_actors_from_string(s: str) -> list[str]:
    tokens = split_multi(s)
    top2 = tokens[:2]
    return [t.replace("_", " ").title() for t in top2]

def normalize_title(t: str) -> str:
    t = nz(t).lower()
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def pretty_genres(g: str) -> str:
    g = nz(g)
    if not g:
        return ""
    g2 = re.sub(r"[;,/]+", "|", g)
    g2 = re.sub(r"\|{2,}", "|", g2)
    parts = [p.strip() for p in g2.split("|") if p.strip()]
    return " | ".join(parts)

# -------------------- Minimal column sanity --------------------
for col in ["title_x", "imdb_rating", "genres", "actors", "story", "release_date"]:
    if col not in movies_df.columns:
        movies_df[col] = ""

movies_df["title_x"] = movies_df["title_x"].apply(nz)
movies_df = movies_df[movies_df["title_x"] != ""].copy()
movies_df = movies_df.drop_duplicates(subset=["title_x"]).reset_index(drop=True)

# -------------------- Clean & feature engineering --------------------
movies_df["story"] = movies_df["story"].apply(nz)
movies_df["genres"] = movies_df["genres"].apply(nz)
movies_df["release_date"] = movies_df["release_date"].apply(nz)
movies_df["actors"] = movies_df["actors"].apply(nz)

movies_df["actors_top2"] = movies_df["actors"].apply(top2_actors_from_string)
movies_df["genres_pretty"] = movies_df["genres"].apply(pretty_genres)

# Combined text for similarity
movies_df["combined_text"] = (
    movies_df["story"] + " " + movies_df["genres"] + " " + movies_df["actors"]
)
movies_df["combined_text"] = movies_df["combined_text"].apply(nz)
movies_df = movies_df[movies_df["combined_text"] != ""].reset_index(drop=True)

# -------------------- Vectorize & Similarity --------------------
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies_df["combined_text"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Index maps
movies_df["title_norm"] = movies_df["title_x"].apply(normalize_title)
idx_by_title = pd.Series(movies_df.index, index=movies_df["title_norm"]).drop_duplicates()
all_norm_titles = movies_df["title_norm"].tolist()

# Flatten all actors for autocomplete (FULL actor list)
all_actors_set = set()
for s in movies_df["actors"]:
    for a in split_multi(s):
        all_actors_set.add(a.lower())
all_actors_list = sorted(list(all_actors_set))

# -------------------- Public API --------------------
def search_titles(query: str, limit: int = 10) -> list[str]:
    q = normalize_title(query)
    if not q:
        return []
    mask = movies_df["title_norm"].str.contains(re.escape(q), na=False)
    hits = movies_df.loc[mask, "title_x"].head(limit).tolist()
    return hits

def search_actors(query: str, limit: int = 10) -> list[str]:
    q = query.lower().strip()
    if not q:
        return []
    matches = [a.title() for a in all_actors_list if q in a]
    return matches[:limit]

def _best_match_index(movie_name: str):
    """Exact match only, no fuzzy"""
    q = normalize_title(movie_name)
    if q in idx_by_title:
        return int(idx_by_title[q])
    return None

def recommend(movie_name: str, top_n: int = 5) -> pd.DataFrame:
    idx = _best_match_index(movie_name)
    if idx is None:
        return pd.DataFrame()
    sims = list(enumerate(cosine_sim[idx]))
    sims = sorted(sims, key=lambda x: x[1], reverse=True)
    picked = []
    for i, _ in sims:
        if i == idx:
            continue
        picked.append(i)
        if len(picked) >= top_n:
            break
    recs = movies_df.iloc[picked].copy()
    out = pd.DataFrame({
        "title": recs["title_x"].values,
        "rating": recs["imdb_rating"].fillna("").values,
        "actors": recs["actors"].apply(lambda s: [a.title() for a in split_multi(s)]).values,
        "genre": recs["genres_pretty"].values,
        "release_date": recs["release_date"].values,
        "story": recs["story"].fillna("").values
    })
    return out

def get_movies_by_actor(actor_name: str) -> pd.DataFrame:
    actor_name = actor_name.lower().strip()
    mask = movies_df["actors"].apply(lambda s: any(actor_name in a.lower() for a in split_multi(s)))
    filtered = movies_df[mask]
    if filtered.empty:
        return pd.DataFrame()
    out = pd.DataFrame({
        "title": filtered["title_x"].values,
        "rating": filtered["imdb_rating"].fillna("").values,
        "actors": filtered["actors"].apply(lambda s: [a.title() for a in split_multi(s)]).values,
        "genre": filtered["genres_pretty"].values,
        "release_date": filtered["release_date"].values,
        "story": filtered["story"].fillna("").values
    })
    return out
