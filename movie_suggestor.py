import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os, urllib.request, zipfile


@st.cache_data
def load_data():
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    zipname = "ml-latest-small.zip"

    if not os.path.exists(zipname):
        urllib.request.urlretrieve(url, zipname)
    if not os.path.exists("ml-latest-small"):
        with zipfile.ZipFile(zipname, "r") as z:
            z.extractall(".")

    movies = pd.read_csv("ml-latest-small/movies.csv")
    ratings = pd.read_csv("ml-latest-small/ratings.csv")

    genre_dummies = movies["genres"].str.get_dummies(sep="|")
    movies = pd.concat([movies, genre_dummies], axis=1)

    movie_stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()
    movie_stats = movie_stats[movie_stats["count"] >= 20]
    movie_stats = pd.merge(movie_stats, movies, on="movieId", how="left")

    scaler = StandardScaler()
    X = scaler.fit_transform(movie_stats[["mean", "count"] + list(genre_dummies.columns)])

    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    movie_stats["Cluster"] = kmeans.fit_predict(X)
    movie_stats.set_index("title", inplace=True)

    return movie_stats



movie_stats = load_data()



st.set_page_config(page_title="🎬 Movie Recommender", layout="centered", page_icon="🎥")

st.markdown(
    """
    <style>
        [data-testid="stAppViewContainer"] {background-color: #121212; color: white;}
        [data-testid="stHeader"] {background: rgba(0,0,0,0);}
        h1, h2, h3, h4 {color: #00b4d8 !important; text-align: center;}
        .stTextInput>div>div>input {
            background-color: #1e1e1e; color: white; border-radius: 8px;
        }
        .stButton>button {
            background-color: #00b4d8; color: white; border-radius: 10px;
            font-size: 16px; padding: 0.6em 1.2em; transition: 0.3s;
        }
        .stButton>button:hover {background-color: #0096c7;}
        div[data-testid="stDataFrame"] {background-color: #1e1e1e !important;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎬 Movie Recommender System")
st.markdown("### Type a movie name to get similar recommendations!")


search_text = st.text_input("🔍 Type to search a movie:")

# Generate suggestions
suggestions = []
if search_text:
    suggestions = [m for m in movie_stats.index if search_text.lower() in m.lower()][:10]

# Show dropdown only if there are matches
movie_name = None
if suggestions:
    movie_name = st.selectbox("🎥 Matching Movies:", suggestions, key="movie_select")
elif search_text:
    st.warning("No movies found! Try another keyword.")


if movie_name and st.button("🎯 Recommend Similar Movies"):
    cluster = movie_stats.loc[movie_name, "Cluster"]
    st.success(f"**'{movie_name}' belongs to Cluster {cluster}** 🎯")

    similar_movies = (
        movie_stats[movie_stats["Cluster"] == cluster]
        .sort_values("mean", ascending=False)
        .drop(movie_name, errors="ignore")
        .head(10)
    )

    st.subheader("🎥 Recommended Movies")
    st.dataframe(
        similar_movies[["mean", "count"]]
        .rename(columns={"mean": "⭐ Avg Rating", "count": "🎟️ Rating Count"})
    )


st.subheader("📊 Cluster Distribution")
st.bar_chart(movie_stats["Cluster"].value_counts().sort_index())
