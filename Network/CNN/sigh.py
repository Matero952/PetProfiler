import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
table = [
    ['Environment', 'Accuracy', 'Precision', 'Recall', 'F1 Score', '         False Positive Rate'],
    ['Fall', '0.9956', '1', '0.9868', '0.9934', '0'],
    ['Snowy', '0.9896', '0.9475', '1', '0.973', '0.01283'],
    ['Varied(Test Split)', '0.9884', '0.9683', '1', '0.9839', '0.01791'],
    ['Grassy', '0.9948', '0.997', '0.9956', '0.9963', '0.006842'],
    ['Average Values', '0.9921', '0.9782', '0.9956', '0.98665', '0.0093955']
]
df = pd.DataFrame(table[1:], columns=table[0])  # Use the first row as the header
df_numeric = df.iloc[:, 1:].apply(pd.to_numeric)  # Convert everything except 'Environment' to numeric

# Create a mask for the heatmap: mask out the first row (header)
mask = np.zeros_like(df_numeric, dtype=bool)

# Create the heatmap
fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(df_numeric, annot=True, fmt='.4f', cmap="coolwarm", cbar=True, mask=mask, ax=ax, annot_kws={"size": 20} )


for i, header in enumerate(df.columns[1:]):
    ax.text(i + 0.5, -0.3, header, ha='center', va='center', fontsize=12, weight='bold')

# # Set labels for the first column manually
# for i, label in enumerate(df['Environment']):
#     ax.text(-0.5, i + 0.5, label, ha='center', va='center', fontsize=12, color="black")

# Remove x and y ticks
ax.set_xticks([])
# Rotate the y-axis labels (row labels)
ax.set_yticklabels(df['Environment'], rotation=45, ha='right', fontsize=11, fontweight="bold")

ax.set_title("Test Results for Key Metrics", fontsize=18, fontweight="bold", pad=40)
plt.show()
