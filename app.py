from flask import Flask, render_template, request, jsonify
import pandas as pd
from movies import (
    recommend, search_titles, get_movies_by_actor,
    search_actors, search_genres, get_movies_by_genre, get_top_rated_movies, movies_df
)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/search")
def search():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify([])

    title_suggestions = search_titles(q)
    actor_suggestions = search_actors(q)
    genre_suggestions = search_genres(q)

    top_suggestions = []
    if q.lower() in ["top", "best", "high", "rating", "high rating", "top 10 rated movies"]:
        top_suggestions = ["Top 10 Rated Movies"]

    suggestions = title_suggestions + actor_suggestions + genre_suggestions + top_suggestions
    return jsonify(suggestions[:10])

@app.route("/recommend", methods=["POST"])
def recommend_movies():
    movie_name = request.form["movie_name"].strip()
    rec_df = None

    # Genre search
    if movie_name.lower().startswith("genre:") or any(g in movie_name.lower() for g in ["action","comedy","horror","drama","thriller"]):
        genre = movie_name.lower().replace("movies","").replace("genre:","").strip()
        rec_df = get_movies_by_genre(genre, limit=10)

    # Top-rated search
    elif movie_name.lower() in ["top 10 rated movies", "top", "best", "high rating"]:
        rec_df = get_top_rated_movies(limit=10)

    # Movie similarity or partial title search
    else:
        rec_df = recommend(movie_name, top_n=5)

        if rec_df.empty:
            partial_mask = movies_df["title_norm"].str.contains(movie_name.lower())
            filtered = movies_df[partial_mask].copy()
            if not filtered.empty:
                filtered["imdb_rating"] = pd.to_numeric(filtered["imdb_rating"], errors="coerce")
                filtered = filtered.sort_values(by="imdb_rating", ascending=False).head(5)
                rec_df = filtered[["title_x", "imdb_rating", "actors", "genres_pretty", "release_date", "story"]]
                rec_df = rec_df.rename(columns={
                    "title_x": "title",
                    "imdb_rating": "rating",
                    "genres_pretty": "genre"
                })

        if rec_df.empty:
            rec_df = get_movies_by_actor(movie_name)

    not_found = rec_df.empty
    recs_list = rec_df.to_dict(orient="records") if not not_found else []

    return render_template(
        "result.html",
        movie_name=movie_name,
        recommendations=recs_list,
        not_found=not_found
    )

if __name__ == "__main__":
    app.run(debug=True)
