#This file is just for making some graphs for the presentaiton.
import matplotlib.pyplot as plt

# Data for Dog and Null categories
labels_outer = ['Dog', 'Null']  # Main sections
sizes_outer = [751 * 3, 2130 * 3]  # Counts for dog and null categories

# Data for sub-categories
# Dog sub-categories
labels_inner_dog = ['Unedited', 'Winter/Fall/Summer-Overlay', 'Street', 'Other']
sizes_inner_dog = [66 * 3, 263 * 3, 158 * 3, 792]

# Null sub-categories
labels_inner_null = ['Empty', 'Human', 'Snake', 'Squirrel/Small Rodents', 'Raccoons', 'Unrecognized-Dogs', 'Unrecognized-Cats']
sizes_inner_null = [223 * 3, 248 * 3, 201 * 3, 397 * 3, 225 * 3, 442 * 3, 394 * 3]

# Create figure
fig= plt.figure(figsize=(21, 10))
fig.suptitle("Dataset Composition After Data Augmentation", size=30)

ax1 = fig.add_axes((0.03, 0.37, 0.5, 0.5))
ax2 = fig.add_axes((0.55, 0.37, 0.35, 0.5))
#
# # Outer pie chart
# ax.pie(sizes_outer, labels=labels_outer, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'black'}, radius=1.35)
# #
# Create the inner pie chart for Dog
colors_4 = ["#FF6B6B",  # Light Red
            "#98FB98",  # Light Green
            "#A2CFFE",  # Pastel Sky Blue
            "#FFD1B3"]  # Peach
colors_8 = ["#FF6B6B",  # Light Red
            "#98FB98",  # Light Green
            "#A2CFFE",  # Pastel Sky Blue
            "#FFD1B3",  # Peach
            "#FF9A8B",  # Coral
            "#B5EAD7",  # Mint Green
            "#E6A8D7",  # Lavender
            "#FFD1DC"]  # Blush Pink
ax1.pie(sizes_inner_dog, labels=labels_inner_dog, autopct='%1.1f%%', radius=1.25,
       wedgeprops={'edgecolor': 'black'}, pctdistance=0.7, colors=colors_4, startangle=180)
for label in ax1.texts:
    label.set_fontsize(19)
    label.set_rotation(0)
    label.set_fontweight("semibold")

# Create the inner pie chart for Null
ax2.pie(sizes_inner_null, labels=labels_inner_null, autopct='%1.1f%%', radius=1.25, startangle=100,
       wedgeprops={'edgecolor': 'black'}, pctdistance=0.7, colors=colors_8)
for label in ax2.texts:
    label.set_fontsize(19)
    label.set_rotation(0)
    label.set_fontweight("semibold")
#

# # Equal aspect ratio ensures that pie is drawn as a circle
# ax.axis('equal')

# Show plot
dog_table_data = [
    ['Subclass', 'Count'],
    ['Unedited', '198'],
    ['Winter/Fall/Summer-Overlay', '789'],
    ['Street','474'],
    ['Other', '792'],
]
null_table_data = [
    ['Subclass', 'Count'],

    ['Empty', '669'],
    ['Human', '744'],
    ['Snake', '603'],
    ['Squirrel/Small Rodents', '1191'],
    ['Raccoons', '675'],
    ['Unrecognized-Dogs', '1326'],
    ['Unrecognized-Cats', '1182'],
]

ax1.axis('off')
ax2.axis('off')
ax1.set_title("Dog Class", size=25, pad=10, weight="light")
ax2.set_title("Null Class", size=25, pad=10, weight="light")
table = ax1.table(cellText=dog_table_data, colLabels=None, loc='center', cellLoc='center', bbox=[-0.2, -0.73, 1.5, 0.6])
table2 = ax2.table(cellText=null_table_data, colLabels=None, loc='center', cellLoc='center', bbox=[-0.2, -0.73, 1.5, 0.6])
table.auto_set_font_size(False)
table2.auto_set_font_size(False)
for (i, j), cell in table.get_celld().items():
    cell.set_fontsize(19)
for (i, j), cell in table2.get_celld().items():
    cell.set_fontsize(19)
# Adjust the space between the plot and the table
plt.subplots_adjust(bottom=0.1)
# plt.title("Dataset Composition After Data Augmentation", size=15, pad=28)
plt.show()
