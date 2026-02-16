import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# --------------------------
# 1. Load dataset
# --------------------------
df = pd.read_csv("songs_processed.csv")

# --------------------------
# 2. TF-IDF + Nearest Neighbors
# --------------------------
tfidf = TfidfVectorizer(analyzer='word', stop_words='english')
matrix = tfidf.fit_transform(df['song'])  # lyrics column
nn = NearestNeighbors(metric="cosine", algorithm="brute")
nn.fit(matrix)

def recommend(song_title, n_recs=5):
    try:
        song_index = df[df['song'].str.lower() == song_title.lower()].index[0]
    except IndexError:
        return []  # Song not found

    distances, indices = nn.kneighbors(matrix[song_index], n_neighbors=n_recs+1)
    recs = []
    for i in range(1, len(indices[0])):  # skip the song itself
        idx = indices[0][i]
        recs.append({
            "title": df.iloc[idx]['song'],
            "artist": df.iloc[idx]['artist'] if 'artist' in df.columns else None
        })
    return recs

# --------------------------
# 3. Spotify API setup
# --------------------------
client_id = "4645457766ca483fb6616b898464ac30"
client_secret = "80cc636368a64a95943803beef505847"

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=client_id,
    client_secret=client_secret
))

def get_album_cover(song_title, artist=None):
    query = song_title if not artist else f"{song_title} {artist}"
    results = sp.search(q=query, limit=1, type='track')
    if results['tracks']['items']:
        return results['tracks']['items'][0]['album']['images'][0]['url']
    return None

# --------------------------
# 4. Streamlit frontend
# --------------------------
st.title("🎶 Song Recommendation System")

song_name = st.text_input("Enter a song title:")

if st.button("Recommend"):
    if song_name:
        recs = recommend(song_name, n_recs=5)
        if not recs:
            st.error("❌ Song not found in dataset.")
        else:
            st.subheader(f"Recommendations for: {song_name}")
            for rec in recs:
                col1, col2 = st.columns([1,3])
                with col1:
                    cover_url = get_album_cover(rec['title'], rec.get('artist'))
                    if cover_url:
                        st.image(cover_url, width=150)
                    else:
                        st.write("No cover found")
                with col2:
                    st.write(f"**{rec['title']}**")
                    if rec.get('artist'):
                        st.write(f"Artist: {rec['artist']}")
