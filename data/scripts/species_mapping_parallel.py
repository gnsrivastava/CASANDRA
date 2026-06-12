# The script maps the species in protein chemical interaction/associations found in STITCH v5

import pandas as pd
from concurrent.futures import ProcessPoolExecutor

def _init_worker(vm):
    global _version_map
    _version_map = vm

def _process_chunk(chunk):
    chunk['Species'] = chunk.protein.str.split('.', expand=True)[0]
    chunk['_version'] = chunk['Species'].map(_version_map)
    return chunk[chunk['string_version'] == chunk['_version']].drop(columns='_version')

if __name__ == '__main__':
    # Load the dataset
    df = pd.read_csv('protein_chemical.links.with_string_version.tsv', sep='\t', chunksize=100_000)

    # Load species data
    Species = pd.read_csv('species.list', sep='\t') # 1st col: specis, 2nd col: versions
    Species['species'] = Species.species.astype('str')

    version_map = Species.set_index('species')['versions']

    hdf_path = 'protein_chemical.links.with_string_version.hdf5'
    with ProcessPoolExecutor(initializer=_init_worker, initargs=(version_map,)) as pool:
        for i, result in enumerate(pool.imap(_process_chunk, df)):
            result.to_hdf(hdf_path, key='data', mode='w' if i == 0 else 'a',
                          append=True, format='table')