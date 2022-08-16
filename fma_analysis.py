import spotipy
from spotipy.oauth2 import SpotifyOAuth
from cred import CLIENT_ID, CLIENT_SECRET, PLAY_LIST, REDIRECT_URL


def getSongs():
    scope = "playlist-read-collaborative"

    #scope = "user-library-read"
    token = SpotifyOAuth(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, redirect_uri=REDIRECT_URL,scope=scope)
    #cache_token = token.get_access_token()
    spotify = spotipy.Spotify(auth=token)

    results = spotify.current_user_playlists()
    for res in results:
        if res == PLAY_LIST:
            return result

    return None