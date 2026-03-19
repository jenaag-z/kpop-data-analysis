import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the data into the variable 'df'
df = pd.read_csv("kpop_cleaned_subset.csv")

# 2. Use 'df' here instead of the file name
corr = df.corr(numeric_only=True)

# 3. Create the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation of Musical Attributes')

# 4. Save it like we did before
plt.savefig('plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')

plt.savefig('plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')

print("Plot successfully saved to the plots folder!")
plt.show()