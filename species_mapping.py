# The script maps the species in protein chemical interaction/associations found in STITCH v5

import pandas as pd

# Load the dataset
df = pd.read_csv('protein_chemical.links.with_string_version.tsv', sep='\t', chunksize=100_000)

# Load species data
Species = pd.read_csv('species.list', sep='\t') # 1st col: specis, 2nd col: versions
Species['species'] = Species.species.astype('str')

# define a final association dataframe
data = pd.DataFrame()

version_map = Species.set_index('species')['versions']

chunks = []
for chunk in df:
    chunk['Species'] = chunk.protein.str.split('.', expand=True)[0]
    chunk['_version'] = chunk['Species'].map(version_map)
    chunks.append(chunk[chunk['string_version'] == chunk['_version']].drop(columns='_version'))

data = pd.concat(chunks, ignore_index=True)
data.to_hdf('protein_chemical.links.with_string_version.hdf5', key=data)