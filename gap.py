# !/usr/bin/python
# -*- coding:utf-8 -*-

import scipy as sp
import scipy.spatial.distance
import numpy as np
from sklearn.cluster import KMeans
import time

dist = scipy.spatial.distance.cosine

def gap(data, run=1, nrefs=20, mink=1, kstep=1, maxlim=False, maxtime=60,file_save_data='Gap_Statistics_report.txt'):
    """
    data is a m x n matrix, where the rows are the observations and the columns the features.
    run is an index that gives marks the "Gapclustering_run_.txt" file
    nrefs is the number of null references. The clusters will be build as follows:
    from mink we increase the number of clusters to be computed in steps of kstep, up to maxlim if maxlim!=0
    or maxlim!=False, or up to a maxtime in minutes. sphere tells whether the data is on an hyper-ball.
    """
    time1 = time.time()
    shape = data.shape
    tops = data.max(axis=0)
    bots = data.min(axis=0)
    dists = sp.matrix(sp.diag(tops-bots))
    rands = sp.random.random_sample(size=(*shape, nrefs))
    for i in range(nrefs):
        rands[:,:,i] = rands[:,:,i]*dists + bots
    gaps = []
    sk = []
    criteria = []
    i = 0
    k = mink
    oldclusters = 0
    old_clusters = []
    oldlabels = 0
    old_labels = []
    numclusters = 0
    num_clusters = []
    while True:
        kmeans = KMeans(n_clusters=k, n_jobs=-1)
        kmeans.fit(data)
        kmc = kmeans.cluster_centers_
        kml = kmeans.labels_
        disp = sum([dist(data[m,:], kmc[kml[m],:])**2/(2*list(kml).count(kml[m])) for m in range(shape[0])])
        del kmeans
        refdisps = sp.zeros( (rands.shape[2],))
        for j in range(rands.shape[2]):
            kmeans = KMeans(n_clusters=k, n_jobs=-1)
            kmeans.fit(rands[:, :, j])
            kmc = kmeans.cluster_centers_
            kml = kmeans.labels_
            refdisps[j] = sum([dist(rands[m,:,j], kmc[kml[m],:])**2/(2*list(kml).count(kml[m])) for m in range(shape[0])])
            del kmeans
        wbar = sp.mean(sp.log(refdisps))
        gaps.append(wbar - sp.log(disp))
        sk.append(sp.std(sp.log(refdisps))*np.sqrt(1+1./nrefs))
        timen = time.time()
        elapsed_time = (timen - time1)/60  # mins
        with open(file_save_data.format(run), 'a') as report:
            report.write("{0} clusters done. Time in mins: {1}\n".format(k, elapsed_time))
        is_crit_gt_0 = False  # is the gap criteria greater than 0?
        if i >= 1:
            new_crit = gaps[i-1] - (gaps[i]-sk[i])
            criteria.append(new_crit)
            if (k >= maxlim or new_crit > 0) and maxlim:
                is_crit_gt_0 = True
                numclusters = k - kstep
        if is_crit_gt_0:
            break
        if elapsed_time >= maxtime:
            break
        oldclusters = kmc
        old_clusters.append(kmc)
        oldlabels = kml
        old_labels.append(kml)
        num_clusters.append(k)
        i += 1
        k += kstep
        print(num_clusters[-1]," ",gaps[-1]," ",sk[-1])
    return num_clusters, old_clusters, old_labels, gaps, sk