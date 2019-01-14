import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tkinter import *
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
# importing data set
rootWindow = Tk()
rootWindow.title("ENISo Hierarchical clustering demo")
rootWindow.configure(background='white')
rootWindow.geometry("500x500")
rootWindow.resizable(0, 0)

headerPhoto = PhotoImage(file='header.png')
label1 = Label(rootWindow, image=headerPhoto, bg='white').place(x=125, y=35)

Label(rootWindow, text="Select data set size: ").pack(side = LEFT)
selectedDataSetValue = StringVar()
selectedDataSetValue.set(100)
dataSetOptions = [100, 200, 500, 1000, 2000]
set1 = OptionMenu(rootWindow, selectedDataSetValue, *dataSetOptions)
set1.pack(side=LEFT)

Label(rootWindow, text="Select the number of clusters desired: ").pack(side=LEFT, padx=(50, 0))
selectedNumOfClustersValue = StringVar()
selectedNumOfClustersValue.set(3)
numOfClustersOptions = [2, 3, 4, 5, 6, 7, 8, 9, 10]
set2 = OptionMenu(rootWindow, selectedNumOfClustersValue, *numOfClustersOptions)
set2.pack(side=LEFT, anchor=W)

Label(rootWindow, text="USED METRIC: EUCLIDEAN DISTANCE", foreground="red").place(x=150, y=280)

Label(rootWindow, text="Select clustering linkage type: ").place(x=130, y=310)
selectedClusteringTypeValue = StringVar()
selectedClusteringTypeValue.set("ward")
clusteringType = ["ward", "complete", "average", "single"]
set3 = OptionMenu(rootWindow, selectedClusteringTypeValue, *clusteringType)
set3.place(x=290, y=305)

# metric is comment calculer la valeur de dist d1, linkage est comment est calculer la dist(max(d1,d2)) etc,
Label(rootWindow, text="NOTE about linkage types\n•ward minimizes the variance of the clusters being merged.\n•average uses the average of the distances of each observation of the two sets.\n•complete linkage uses the maximum distances between all observations of the two sets.\n•single uses the minimum of the distances between all observations of the two sets.", foreground="red").place(x=12, y=350)




############################## CODE STARTS


dataset = pd.read_csv('data.csv')


def drawClusters():

    x = dataset.iloc[2000: 2000 + int(selectedDataSetValue.get()), [3, 5]].values

    # using dendogram to find the optimal number of clusters

    # ward = minimum of distances
    dendrogram = sch.dendrogram(sch.linkage(x, method=selectedClusteringTypeValue.get()))
    # fitting hierarchical clustering to the data set
    print("method="+selectedClusteringTypeValue.get())
    print("datasetsize="+selectedDataSetValue.get())
    print("n_clusters="+selectedNumOfClustersValue.get())
    hc = AgglomerativeClustering(n_clusters=int(selectedNumOfClustersValue.get()), affinity='euclidean', linkage=selectedClusteringTypeValue.get())
    y_hc = hc.fit_predict(x)

    # visualise clusters
    for i in range(0, int(selectedNumOfClustersValue.get())):
        plt.scatter(x[y_hc == i, 0], x[y_hc == i, 1], s=10, label='Cluster'+str(i+1))

    # plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s=5, label='Cluster 2')
    # plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s=5,  label='Cluster 3')
    # plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s=5, label='Cluster 4')
    # plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s=5,  label='Cluster 5')
    # plt.scatter(x[y_hc == 5, 0], x[y_hc == 5, 1], s=5, label='Cluster 6')

    plt.title('Clustering of bought items')
    plt.xlabel('Quantity')
    plt.ylabel('Item price')
    plt.legend()
    plt.show()


Button(rootWindow, text="Start clustering", command=drawClusters).place(x=200, y=450)
rootWindow.mainloop()

