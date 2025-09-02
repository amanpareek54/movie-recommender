from flask import Flask, render_template, request, jsonify
from movies import recommend, search_titles, get_movies_by_actor, search_actors

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/search")
def search():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify([])
    
    title_suggestions = search_titles(q)
    actor_suggestions = search_actors(q)
    suggestions = title_suggestions + actor_suggestions
    return jsonify(suggestions[:10])

@app.route("/recommend", methods=["POST"])
def recommend_movies():
    movie_name = request.form["movie_name"].strip()

    # Exact movie title recommendations only
    rec_df = recommend(movie_name)

    # If no movie match, try actor search
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
