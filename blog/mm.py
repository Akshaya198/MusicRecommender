import pandas as pd
import numpy as np

song_data=pd.read_csv('C:/Users/Akshaya Lakshmi/dj/blog/SpotifyFeatures.csv')

song_data.head()

song_data.sort_values('track_name')

song_data.count()

song_data.duplicated('track_name')

ss=song_data

ss.count()

ssclean=ss.drop_duplicates(subset='track_name')

ssclean.count()

ssclean.head(n=20)

ssclean=ssclean.sort_values('track_name')

ssclean.head()

ssclean.tail()

spop=ssclean.sort_values('popularity')

spop.tail(n=10)

def knnQuery(queryPoint, arrCharactPoints, k):
    tmp = arrCharactPoints.copy(deep=True)
    tmp['dist'] = tmp.apply(lambda x: np.linalg.norm(x-queryPoint), axis=1)
    tmp = tmp.sort_values('dist')
    return tmp.head(k).index

def querySimilars(df, columns, idx, func, param):
    arr = df[columns].copy(deep=True)
    queryPoint = arr.loc[idx]
    arr = arr.drop([idx])
    response = func(queryPoint, arr, param)
    return response

def getMusicName(elem):
    return '{} - {}'.format(elem['artist_name'], elem['track_name'])

def model_(artist,music):
    a=[]
    row_num = spop[(spop['track_name'] == music) & (spop['artist_name'] == artist)].index.to_numpy() 
    songIndex = row_num[0] # query point, selected song
    columns = ['acousticness','danceability','energy','instrumentalness','liveness','speechiness','valence']
# Selecting query parameters
    func, param = knnQuery, 3 # k=3
# Querying
    response = querySimilars(spop, columns, songIndex, func, param)

    anySong = spop.loc[songIndex]
# Get the song name
    anySongName = getMusicName(anySong)
# Print
    print('#Query Point')
    print(songIndex, anySongName)

    print('# Similar songs')
    for idx in response:
        anySong = spop.loc[idx]
        anySongName = getMusicName(anySong)
        print(idx, anySongName)
        a.append(anySongName)
    return a

def pop(artist,music):
    y=spop[(spop['track_name'] == music) & (spop['artist_name'] == artist)].popularity
    z=y.iloc[0] 
    return z