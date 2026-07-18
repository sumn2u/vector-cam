import pandas as pd
import matplotlib.pyplot as plt
import glob
import os


folder_path = "csvs/"   
all_dfs = []

for file in glob.glob(os.path.join(folder_path, "*.csv")):
    df = pd.read_csv(file)
    all_dfs.append(df)

# Combine all data into one DataFrame
combined = pd.concat(all_dfs, ignore_index=True)

counts = combined.groupby(['SpeciesLabel', 'DeviceModel']).size().reset_index(name='SpecimenCount')

pivot_table = counts.pivot(index='SpeciesLabel', columns='DeviceModel', values='SpecimenCount').fillna(0)

ax = pivot_table.plot(kind='bar', figsize=(14, 7), width=0.8)

# Customize the chart
plt.title('Specimen Counts by Species and Device Model', fontsize=16)
plt.xlabel('Species', fontsize=12)
plt.ylabel('Number of Specimens', fontsize=12)
plt.xticks(rotation=45, ha='right')   # rotate species names for readability
plt.legend(title='Device Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Show the plot
# plt.show()

plt.savefig('specimen_counts_by_species.png', dpi=300)