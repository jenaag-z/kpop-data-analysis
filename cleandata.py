import pandas as pd

# 1. Load the raw Kaggle dataset
# Make sure the file name matches your downloaded file exactly
try:
    df = pd.read_csv("C:\\Users\\DIL\\Downloads\\archive\\spotify-tracks-dataset.csv")
except FileNotFoundError:
    print("Error: Could not find 'spotify_tracks.csv' in the data folder.")

# 2. Filter for Stray Kids and ATEEZ
# We use a case-insensitive search in the 'artists' column
kpop_filter = df['artists'].str.contains('Stray Kids|ATEEZ', case=False, na=False)
kpop_subset = df[kpop_filter].copy()

# 3. Handle Duplicates
# Keep the version of the song with the highest popularity
kpop_subset = kpop_subset.sort_values('popularity', ascending=False)
kpop_cleaned = kpop_subset.drop_duplicates(subset='track_name', keep='first')

# 4. Select the features needed for K-Means Clustering
features = [
    'track_id', 'artists', 'track_name', 
    'danceability', 'energy', 'loudness', 'speechiness', 
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]
final_df = kpop_cleaned[features]

# 5. Save the cleaned subset
final_df.to_csv("kpop_cleaned_subset.csv", index=False)

print(f"Success! Created kpop_cleaned_subset.csv with {len(final_df)} tracks.")

# Load your new small dataset
df_kpop = pd.read_csv("kpop_cleaned_subset.csv")

# 1. Check for missing values in audio features
print("Missing values per column:")
print(df_kpop.isnull().sum())

# 2. Check the average 'energy' and 'danceability' for both groups
summary = df_kpop.groupby('artists')[['energy', 'danceability', 'tempo']].mean()
print("\nQuick Musical Profile:")
print(summary)