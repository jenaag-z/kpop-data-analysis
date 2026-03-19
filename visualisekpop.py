import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# create folder for plots
if not os.path.exists('plots'):
    os.makedirs('plots')

# Load the clean data
df = pd.read_csv("kpop_cleaned_subset.csv")

# Create a scatter plot of Energy vs. Danceability
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='danceability', y='energy', hue='artists', s=100)

plt.title('Energy vs. Danceability: Stray Kids & ATEEZ')
plt.grid(True, linestyle='--', alpha=0.6)

# SAVE THE PLOT
# dpi=300 makes it high resolution for your GitHub README
plt.savefig('plots/energy_vs_danceability.png', dpi=300, bbox_inches='tight')
plt.show()