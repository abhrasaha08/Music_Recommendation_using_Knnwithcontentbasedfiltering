import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
# Load the dataset (replace with your actual path)
data = pd.read_csv('songsforproject.csv')
data = data.drop(columns=['Unnamed: 2']) #to drop the unnammed or deleted the column
pd.set_option('display.width', 1000)  # Adjust width as needed
pd.set_option('display.max_columns', None)  # Show all columns without truncation

data['Song_Name'] = data['Song_Name'].str.strip()
# Normalize the features (Tempo, Energy, Danceability, Acousticness)
scaler = MinMaxScaler()
data[['Tempo', 'Energy', 'Danceability', 'Acousticness']] = scaler.fit_transform(data[['Tempo', 'Energy', 'Danceability', 'Acousticness']])
print(data.head())
# Encode the categorical Song_Type (Emotion)
label_encoder = LabelEncoder()
data['Song_Type'] = label_encoder.fit_transform(data['Song_Type'])
print(data.head())
# Independent variables (features)
X = data[['Tempo', 'Energy', 'Danceability', 'Acousticness']]

# Dependent variable (target)
y = data['Song_Type']  # Not strictly needed for KNN content-based filtering, but useful for filtering

# Check the split
print(X.head())  # Features
print(y.max())  # Target (Song_Type)
# Fit the KNN model
knn = NearestNeighbors(n_neighbors=10)
knn.fit(X)

# Select a song to recommend based on its index
song_index = 110 # Change this index as needed

# Convert the selected song to a DataFrame with proper column names
song_to_recommend = pd.DataFrame([X.iloc[song_index].values], columns=X.columns)

# Get the nearest neighbors (including the song itself)
distances, indices = knn.kneighbors(song_to_recommend)

# Get recommended songs excluding the song itself
recommended_songs = data.iloc[indices[0]]  # Exclude the first song (it will be the same song)

# Display the recommended songs
# Filter the recommended songs based on the Song_Type (emotion)
recommended_songs_filtered = recommended_songs[recommended_songs['Song_Type'] == data.iloc[song_index]['Song_Type']]
#ensuring the type of the song matches here
print("Filtered recommended songs:")
print(recommended_songs_filtered[['Song_Name', 'Artist_Name', 'Song_Type']])
  # Ensure column names match


# Assuming 'recommended_songs' contains recommended songs
plt.figure(figsize=(8,6))
sns.scatterplot(data=data, x='Energy', y='Danceability', alpha=0.5, label="All Songs")
sns.scatterplot(data=recommended_songs, x='Energy', y='Danceability', color='red', label="Recommended Songs")

plt.xlabel("Energy")
plt.ylabel("Danceability")
plt.title("Comparison of Recommended Songs vs All Songs")
plt.legend()
plt.show()
