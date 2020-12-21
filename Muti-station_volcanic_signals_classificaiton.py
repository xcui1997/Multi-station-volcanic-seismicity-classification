#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@file        :Muti-station_volcanic_signals_classificaiton.py
@note        :
@time        :2020/12/05 10:54:53
@author        :xcui
@version        :1.0
'''

import os
import json
import glob
import math
import pickle
import numpy as np
import pandas as pd
import argparse as ap
import matplotlib.pyplot as plt
from obspy import read
from obspy.core import UTCDateTime
from obspy.signal.trigger import recursive_sta_lta
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage


####################################################
"""
This part is data processing.
"""
def SNR(data, arr, rate, win, tol=0.5):
    narr = int(arr * rate)
    nwin = int(win * rate)
    ntol = int(tol * rate)
    noise_e = np.sum(data[narr-nwin:narr-ntol]**2)+1e-15
    signal_e = np.sum(data[narr+ntol:narr+nwin]**2)+1e-15
    return 10*math.log10(signal_e/noise_e)


def amp_nmlz(fz,x):
    fz_nmlz = fz / fz[-1]
    x_int = np.sum(x) * (fz[1]-fz[0])
    x_nmlz = x / x_int *fz[-1]
    return x_nmlz, fz_nmlz


#sometimes the rate may not 100 , so it best for us to have a resample firstly.
#fft
def fft_amp(Data, rate, arr, win_len):
    narr = int(arr * rate)
    nwin_len = int(win_len * rate)
    Data = Data[narr-nwin_len:narr+nwin_len]
    sp = np.fft.fft(Data)
    freq = np.fft.fftfreq(len(Data), d=1.0/rate)
    amp = np.abs(sp)
    A_upper = np.mean(amp[(freq>5.0) & (freq<15.0)])
    A_lower = np.mean(amp[(freq>1.0) & (freq<5.0)])
    return amp[(freq>=0) & (freq<15.0)], freq[(freq>=0) & (freq<15.0)], math.log10(A_upper/A_lower)


def picker(data, rate, nsta=10, nlta=100, uncer=0.5):
    stalta = recursive_sta_lta(data, nsta, nlta)
    n_max = np.argmax(stalta)
    quality = stalta[n_max]
    n_onset = int(uncer * rate)
    n_diff_max = np.argmax(np.diff(stalta[n_max-n_onset:n_max+n_onset]))
    n_pick = n_max - n_onset + n_diff_max
    arr = n_pick / rate
    return  arr, quality

################################################################################################
def remove_dict(dict_event, dict_sta, namp_len):
    need_delete = []
    for key in dict_event.keys():
        num_sta = 0
        for i in dict_sta.values():
            if (dict_event[key][i] != []):
                # remove those which length is not 39
                if len(dict_event[key][i][0])== namp_len:
                    num_sta += 1
        if num_sta < 5:
            need_delete.append(key)
               
    for rem in range(len(need_delete)):
        dict_event.pop(need_delete[rem])
    print("the counts of events that meet the requirements are:\n"+str(len(dict_event.keys())))
    return dict_event


def get_median_value(dict_event, namp_len, whether_plot):
    median_FI = []
    median_amp = []
    events_key = []
    for key, value1 in dict_event.items():
        tempo_FI = []
        tempo_amp = []
        events_key.append(key)
        for para in dict_sta.values():
            if (value1[para] != []):
                if (len(value1[para][0]) == namp_len):
                    tempo_FI.append(value1[para][1])
                    tempo_amp.append(value1[para][0])
        if len(tempo_FI)%2 == 1:
            median_as = tempo_FI.index(np.median(tempo_FI))
            if isinstance(median_as, int):
                median_FI.append(tempo_FI[median_as])
                median_amp.append(tempo_amp[median_as])
            else:
                median_as = median_as[0]
                median_FI.append(tempo_FI[median_as])
                median_amp.append(tempo_amp[median_as])
        else:
            tempo_FI.pop(0)
            median_as = tempo_FI.index(np.median(tempo_FI))
            if isinstance(median_as, int):
                median_FI.append(tempo_FI[median_as])
                median_amp.append(tempo_amp[median_as])
            else:
                median_as = median_as[0]
                median_FI.append(tempo_FI[median_as])
                median_amp.append(tempo_amp[median_as])
    bins_ = np.linspace(-1.5, 1, 30)
    plt.hist(median_FI, bins=bins_, edgecolor='k')
    plt.savefig('out/png/median_FI.png', format = 'png')
    pickle.dump(np.asarray(events_key), open('out/text/events_key.pkl', 'wb'))
    pickle.dump(np.asarray(median_amp), open('out/text/median_amp.pkl', 'wb'))
    if whether_plot:
        plt.show()
    plt.close()
    return median_FI, median_amp, events_key


def calculate_EM(median_amp):
    sum_amp = np.sum(median_amp[0]) * np.sqrt(2)
    median_distance = np.zeros((len(median_amp), len(median_amp)))
    for i in range(len(median_amp)):
        for j in range(i+1,len(median_amp)):
            median_distance[i, j] = np.sqrt(np.sum((median_amp[i][:]-median_amp[j][:])**2))/sum_amp
            median_distance[j, i] = median_distance[i, j]
            
    pickle.dump(np.asarray(median_distance), open('out/text/median_distance.pkl', 'wb'))
    return median_distance

######################################################################
"""
This part is clustering result display
"""
def clust_stats(clust, whether_plot):
    labels = np.unique(clust.labels_)
    n_clusters = len(labels)
    counts = np.zeros(n_clusters, dtype='int')
    for i in range(n_clusters):
        idx = np.where(clust.labels_ == labels[i])[0]
        counts[i] = len(idx)

    plt.figure()
    plt.bar(np.arange(n_clusters), counts)
    for i in range(n_clusters):
        plt.text(i, counts[i], str(counts[i]),
                    horizontalalignment='center',
                    verticalalignment='bottom')
    plt.xticks(np.arange(n_clusters))
    title = 'out/png/hist.png'
    plt.savefig(title, format='png')
    if whether_plot:
        plt.show()
    plt.close()


def new_catalog(labels, event_info, events_key, median_FI, sort_idx):
    k = 1
    new_events_catalog = []
    event_info.values[0][0] = event_info.values[0][0]  \
                              + ' FI' + ' CLUSTER'
    for i in range(len(events_key)):
        for j in range(k, len(event_info.values)):
            if events_key[i] == event_info.values[j][0].split(' ')[0]:
                event_info.values[j][0] = event_info.values[j][0] + ' ' \
                    + str(round(median_FI[i], 2)) + ' ' + str(np.where(sort_idx == labels[i])[0][0])
                new_events_catalog.append(event_info.values[j][0])
                k = j
                break

    for i in range(len(sort_idx)):
        idx_cls = np.where(labels == sort_idx[i])[0]
        with open('./out/text/cluster'+str(i)+'.dat', 'w') as f:
            for j in range(len(idx_cls)):
                f.write(new_events_catalog[idx_cls[j]] + '\n')

    event_info.to_csv('out/text/new_catalog', index=False, header=None)
                 

def plot_rep(clust, amp, sort_idx):
    labels = np.unique(clust.labels_)
    num_rep = 100
    mx, my = 10, 10
    amp = np.array(amp)
    for label in labels:
        idx = np.where(clust.labels_ == sort_idx[label])[0]
        if len(idx)>num_rep:
            sel_idx = idx[np.random.choice(len(idx), num_rep)]
            # the num of every category <= 100 
        else:
            sel_idx = idx
        fig = plt.figure(figsize=(10, 10))
        for i in range(len(sel_idx)):
            plt.subplot(mx, my, i+1)
            plt.plot(amp[idx[i], :]/np.max(amp[idx[i], :]), c='k', linewidth=2)
            plt.axis('off')
        title1 = 'Cluster #' + str(label)
        title2 = ' ('+str(len(idx))+' members)'
        fig.suptitle(title1+title2, fontsize=40)
        title = 'out/png/cls'+str(label)+'.pdf'
        plt.savefig(title, format='pdf')
        plt.close()

#turn it to two function. 1.get the sirt_idx; 2. plot the median spectra
def freq_sort(labels, amp, n_cls):
    amp = np.array(amp)
    mean_amp_container = []
    for i in range(n_cls):
        idx_cls = np.where(labels == i)[0]
        mean_amp = np.zeros(amp.shape[1])
        for j in idx_cls:
            mean_amp += amp[j,:]

        mean_amp = mean_amp / len(idx_cls)
        mean_amp_container.append(mean_amp)
    
    max_amp_idx = np.argmax(mean_amp_container, axis=-1)
    sort_idx = np.argsort(max_amp_idx)
    return np.array(mean_amp_container), sort_idx




def plot_mean_spectra(labels, amp, median_distance, mean_amp_container, sort_idx):
    len_idx = []
    amp = np.array(amp)
    plt.figure(figsize=(16, 10))
    for i in range(len(sort_idx)):
        # retrieve the cluster members
        idx_cls = np.where(labels == sort_idx[i])[0]
        len_idx.append(len(idx_cls))
        X_cls = median_distance[idx_cls, :][:, idx_cls]
        # find the reference smediantf (closest to the cls center)
        # i.e., the one with min  distance from other group members
        median_dist = np.median(X_cls, axis=1)
        imin = np.argmin(median_dist) # locade of the min 
        # obtain and plot the stretched stf relative to the reference
        ax = plt.subplot(4, 5, i+1)
        plt.plot(amp[idx_cls][imin], linewidth=2, c='#333333')
        plt.plot(mean_amp_container[sort_idx[i]], linewidth=2, c='#FF3333')
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.text(0.01, 0.99, 'Cluster #'+str(i),
                fontsize=15, fontweight='bold',
                horizontalalignment='left',
                verticalalignment='center',
                transform=ax.transAxes)
        plt.text(0.99, 0.99, '('+str(len(idx_cls))+')',
                horizontalalignment='right',
                verticalalignment='center',
                fontsize=15, transform=ax.transAxes)
    plt.savefig('out/png/mean_spectra.pdf', format='pdf')
    if whether_plot:
        plt.show()
    plt.close()
    return len_idx


def freq_energy_distribution(mean_amp_container, freq, sort_idx, len_idx):
 # calculate features and plot
    max_amp = np.max(mean_amp_container[sort_idx], axis=-1)
    max_amp_idx = np.argmax(mean_amp_container[sort_idx], axis=-1)
    peak_freq = freq[max_amp_idx]
    plt.scatter(peak_freq, 1./max_amp, alpha = 0.75)
    for i in range(mean_amp_container.shape[0]):
        plt.text(peak_freq[i], 1./max_amp[i], str(i), size =8)
        plt.ylabel('Mean amp/Peak amp')
        plt.xlabel('Peak frequency /HZ')
    plt.savefig('out/png/fre_energy.pdf', format='pdf')
    if whether_plot:
        plt.show()
    plt.close()   

    np.savetxt("out/text/peak_amp_size", list(zip(peak_freq, 1./max_amp, 0.1*np.log10(np.array(len_idx)), np.arange(20))))



def plot_matrix(median_distance):
    plt.figure(figsize=(10, 10))
    #imshow don't support float16
    im = plt.imshow(median_distance, origin='lower', cmap='RdBu',
               vmin=0, vmax=0.25)        
    #imshow don't support float16
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Events', fontsize=25)
    plt.ylabel('Events', fontsize=25) 
    cb = plt.colorbar(im, ticks=[0, 0.05, 0.1, 0.15, 0.2, 0.25])
    cb.set_label('Dissimilarity', fontsize=25) 
    plt.savefig('out/png/med_dis.pdf', format = 'pdf')
    if whether_plot:
        plt.show()
    plt.close()


def plot_dendrogram(median_distance, n_cls):

    X_vec = squareform(median_distance) # The square matrices of vectors convert to each other
    linkage_matrix = linkage(X_vec, "complete")  # Hierarchical clustering
    plt.figure(figsize=(20, 10))
    dendrogram(linkage_matrix, p=n_cls, truncate_mode="lastp")
    ax = plt.gca()
    ax.tick_params(axis='x', which='major', labelsize=20)
    ax.tick_params(axis='y', which='major', labelsize=20)
    # plt.ylabel('amp distance', fontsize= 40)
    # plt.title("Dendrogram with "+str(n_cls)+" clusters", fontsize=40)
    plt.savefig('out/png/dendrogram.pdf', format='pdf')
    if whether_plot:
        plt.show()
    plt.close()
###################################################################


if __name__ == '__main__':
    #get para from config_json
    parser = ap.ArgumentParser(
        prog='Muti-station_volcanic_signals_classification.py',
        description='classified volcano signals')
    parser.add_argument('config_json')
    parser.add_argument(
        '-P',
        default=False,
        action='store_true',
        help='Plot output')
    args = parser.parse_args()
    whether_plot = args.P

    with open(args.config_json, "r") as f:
        params = json.load(f)

    # make dirc
    if not os.path.exists('out/png'):
        os.makedirs('out/png')
    if not os.path.exists('out/text'):
        os.makedirs('out/text')


    dict_sta = {}
    dict_event = {}
    win_snr = 2
    win_len = params["win_len"]
    snr = params["snr"]
    least_station = params["least_station"]
    station_list = params["station_list"]
    events_catalog = params["events_catalog"]
    #the length of amplitude
    namp_len = math.ceil(2*win_len*15)

    
    station_info = pd.read_table(station_list, header=None)
    for i in range(len(station_info.values)):
        dict_sta[station_info.values[i][0]] = i
    event_info = pd.read_table(events_catalog, header=None)
    for i in range(1, len(event_info.values)):
        dict_event[event_info.values[i][0].split(' ')[0]] =  [[] for i in range(len(station_info.values))]
    
    for name, param in dict_sta.items():
        print(name, param)
        for sacname in glob.glob("Hawaii_2020"+name+"/*.SAC"):
            tr = read(sacname)[0]
            t = np.arange(tr.stats.npts) / tr.stats.sampling_rate
            if t[-1] < 10:
                continue

            tr.filter(type='bandpass', freqmin=1, freqmax=15)
            rate = tr.stats.sampling_rate
            datafull = tr.data
            Data = datafull[100:-100]
            p_arr, quality = picker(Data, rate)

            if p_arr > win_snr and p_arr < t[-1]-win_snr and quality > snr:
                snr_value = SNR(Data, p_arr, rate, win_snr)

                if snr_value > snr:

                    amp, freq, fi = fft_amp(Data, rate, p_arr, win_len)
                    Amp_nmlz, freq_nmlz = amp_nmlz(freq, amp)
                    # Put the events that meet the conditions into the dictionary
                    name_event = '-'.join(sacname.split('.')[5:8])
                    name_event = '-'.join(name_event.split('-')[0:3])
                    name_event = str(UTCDateTime(name_event) + 5)
                    if (name_event in dict_event.keys()):
                        #get the amp and the FI value
                        dict_event[name_event][param].append(Amp_nmlz)
                        dict_event[name_event][param].append(fi)
                    # else:
                    #     #it's better to open a file and write them in it.
                    #     print(name+' '+name_event+'not in the catalog')
###########################################################################
    #calculate
    dict_event = remove_dict(dict_event, dict_sta, namp_len)
    median_FI, median_amp, events_key = get_median_value(dict_event, namp_len, whether_plot)
    median_distance = calculate_EM(median_amp)
    plot_matrix(median_distance)
    plot_dendrogram(median_distance, params["n_cls"])
    #Hierarchical clustering
    clust = AgglomerativeClustering(n_clusters=params["n_cls"],
                                    linkage='complete',
                                    affinity='precomputed').fit(median_distance)
    #dendrogram
    clust_stats(clust, whether_plot)
    # plot mean spectra
    mean_amp_container, sort_idx = freq_sort(clust.labels_, median_amp, params["n_cls"])
    len_idx = plot_mean_spectra(clust.labels_, median_amp, median_distance, mean_amp_container, sort_idx)
    #plot 100 spectra
    plot_rep(clust, median_amp, sort_idx)
    # plot freq-evengy space
    freq_energy_distribution(mean_amp_container, freq, sort_idx, len_idx)
    #save files
    new_catalog(clust.labels_, event_info, events_key, median_FI, sort_idx)











