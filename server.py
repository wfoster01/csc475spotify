from flask import Flask
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from flask import request, jsonify
import pandas as pd

def cluster_songs_by_artist(df, artist_name, n_clusters=3):
    # Filter the dataframe to include only songs by the specified artist
    artist_songs = df[df['artists'].apply(lambda x: x.lower() == artist_name.lower())]

    # Select the song features for clustering
    song_features = artist_songs[['acousticness', 'danceability', 'energy', 'instrumentalness', 'speechiness', 'valence']]

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(song_features)

    # Add the cluster labels to the dataframe
    artist_songs['cluster'] = kmeans.labels_

    # Initialize a list to store top songs from each cluster
    top_songs = []

    # Iterate through each cluster and find the top song based on popularity
    for cluster_label in artist_songs['cluster'].unique():
        cluster_group = artist_songs[artist_songs['cluster'] == cluster_label]
        
        # Find the most popular song in this cluster
        top_song = cluster_group.loc[cluster_group['popularity'].idxmax()]

        # Add the artist's name to the top song data
        top_song['artist'] = artist_name

        top_songs.append(top_song)

    # Convert the list of top songs to a DataFrame
    top_songs_df = pd.DataFrame(top_songs)

    return artist_songs[['track_name', 'cluster', 'popularity']], top_songs_df[['track_name', 'cluster', 'popularity', 'artist']]

def find_closest_songs_for_artists(spotify_data, artist_list, top_list):
    for artist_name in artist_list:
        # Filter songs by the current artist
        artist_songs = df[df['artists'] == artist_name]
        # Find the top songs in the original dataset from top_list
        artist_songs = artist_songs[artist_songs['track_name'].isin(top_list[top_list['artist'] == artist_name]['track_name'])]
        artist_song_features = artist_songs[['popularity', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'speechiness', 'valence', 'track_genre']]

        # Filter non-artist songs
        non_artist_songs = spotify_data[spotify_data['artists'] != artist_name]
        non_artist_song_features = non_artist_songs[['popularity', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'speechiness', 'valence', 'track_genre']]

        # Calculate the Euclidean distance between each non-artist song and each song by the current artist
        distances = cdist(non_artist_song_features, artist_song_features, metric='euclidean')

        closest_songs_info = []
        for i in range(len(artist_songs)):
            closest_song_index = distances[:, i].argmin()
            closest_song = non_artist_songs.iloc[closest_song_index]
            closest_song_info = {'track_name': closest_song['track_name'], 'artist': closest_song['artists']}
            closest_songs_info.append(closest_song_info)

        # Print the non-artist song closest to each song by the current artist along with the artist name
        artist_songs_list = artist_songs['track_name'].tolist()
        for i in range(len(artist_list)):
            closest_song_info = closest_songs_info[i]
            print(f"Non-{artist_name} song closest to '{artist_songs_list[i]}': '{closest_song_info['track_name']}' by {closest_song_info['artist']}")

df = pd.read_csv('dataset.csv')
df = df.dropna()

# Make the track_genre column numeric
df['track_genre'] = pd.Categorical(df['track_genre'])
df['track_genre'] = df['track_genre'].cat.codes


app = Flask(__name__)

@app.route("/recommend")
def hello_world():

    artists = []
    # Get three artists from the user

    first  = request.args.get('artist1')
    second = request.args.get('artist2')
    third  = request.args.get('artist3')

    inputs = [first, second, third]

    # while len(artists) < 3:
    for artist in inputs: 
        # artist = input("Enter an artist: ")
        
    # Check if the artist is in the dataframe
        if (df['artists'].str.lower() == artist.lower()).any():
            # Check to see if the artist has at least three unique song names
            if df[df['artists'].str.lower() == artist.lower()]['track_name'].nunique() < 3:
                print("The artist does not have at least three unique songs. Please enter another artist.")
            else:
                # Find the original artist name in the dataset and append it to the list of artists
                artists.append(df[df['artists'].str.lower() == artist.lower()]['artists'].values[0])

        else:
            print("The artist is not in the dataset. Please enter another artist.")
        
    for artist in artists:
        print(artist.title())

    # Initialize an empty DataFrame to store top songs from all artists
    all_top_songs = pd.DataFrame()

    # Loop through the list of artists and find top songs from each cluster
    for artist in artists:
        clustered_songs, top_songs = cluster_songs_by_artist(df, artist.title())
        # If a song is in the top songs of more than one cluster, remove duplicates
        top_songs = top_songs.drop_duplicates(subset=['track_name'])
        print(f"Clustered songs for {artist}:\n{clustered_songs}\n")
        
        # Append the top songs from this artist to the continuous DataFrame
        all_top_songs = pd.concat([all_top_songs, top_songs], ignore_index=True)    

    # Display the continuous DataFrame of top songs from all artists
    print("All top songs from each cluster of each artist:\n", all_top_songs)

    print(all_top_songs)



    spotify_data, artist_list, top_list = df, all_top_songs['artist'].unique(), all_top_songs

    audio_features = ['popularity', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'valence']
    
    result = {}
    for artist_name in artist_list:

        # Filter songs by the current artist
        artist_songs = df[df['artists'] == artist_name]
        top_song_names = all_top_songs[all_top_songs['artist'] == artist_name]['track_name'].values

        top_songs_w_features = artist_songs[artist_songs['track_name'].apply(lambda x: x in top_song_names)]
        top_songs_w_features = top_songs_w_features.groupby('track_name')[audio_features].mean()

        # Find the top songs in the original dataset from top_list
        # artist_songs = artist_songs[artist_songs['track_name'].isin(top_list[top_list['artist'] == artist_name]['track_name'])]
        # artist_song_features = artist_songs[['popularity', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'valence']]

        artist_song_features = top_songs_w_features


        # Filter non-artist songs
        non_artist_songs = spotify_data[spotify_data['artists'] != artist_name]
        non_artist_song_features = non_artist_songs[['popularity', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'valence']]

        # Calculate the Euclidean distance between each non-artist song and each song by the current artist
        distances = cdist(non_artist_song_features, artist_song_features, metric='euclidean')

        closest_songs_info = []
        for i in range(len(artist_song_features)):
            closest_song_index = distances[:, i].argmin()
            closest_song = non_artist_songs.iloc[closest_song_index]
            closest_song_info = {'track_name': closest_song['track_name'], 'artist': closest_song['artists']}
            closest_songs_info.append(closest_song_info)


        artist_songs_list = artist_song_features.index.tolist()
        for i in range(len(artist_song_features)):
            closest_song_info = closest_songs_info[i]
            # print(f"Non-{artist_name} song closest to '{artist_songs_list[i]}': '{closest_song_info['track_name']}' by {closest_song_info['artist']}")
            result[f"Non-{artist_name} song closest to {artist_songs_list[i]}"] = f"{closest_song_info['track_name']} by {closest_song_info['artist']}"

    # return str(result)
    response = jsonify(result)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response
