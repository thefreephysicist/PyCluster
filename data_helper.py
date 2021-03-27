import pandas as pd

def import_data(filename):
    return pd.read_csv(filename, sep='\t', header=None)