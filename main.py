from clusterer import Clusterer
import numpy as np
import data_helper as dh
import hdbscan

def main():
    c = Clusterer()
    detect_pd = dh.import_data(r'D:\svn\Development\trunk\bin\detectresults.tsv')

    # get only first two columns of data for clustering
    data = detect_pd.iloc[:, [0,1]].to_numpy()
    c.set_data(data)
    c.cluster_hdbscan()

    for clusterId in range(1,c.n_clusters+1):
        indices = c.get_cluster(clusterId)
        for i in indices:
            print(data[i])
        print("")

if __name__ == "__main__":
    main()