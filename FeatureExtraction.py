import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neo
import os
from scipy import stats, signal

### Import spike train ###
def load_spiketrain(filepath, to_plot = False):
    reader = neo.io.Spike2IO(filepath)
    block = reader.read(lazy=False)[0]
    segments = block.segments[0]

    analogsignal = np.array(segments.analogsignals[0],dtype='float64').transpose()[0] # raw analog waveform (unfiltered)
    spike_times = np.array(segments.events[0],dtype='float64') # spike timing array (i.e. exact time reletive to the start of the recording when a spike fired an action potential) 
    sampling_frequency = float(segments.analogsignals[0].sampling_rate) # the number of samples per second (in Hz)
    time = np.arange(0,len(analogsignal))/sampling_frequency # time vector
    
    if to_plot == True:
        fig, ax = plt.subplots(2, sharex = True, )
        fig.suptitle(filepath.split('/')[-1], fontsize=16)    

        ax[0].plot(time,analogsignal,'green')
        ax[1].eventplot(spike_times, color='black')
        ax[1].set_xlabel("Time (s)")
    
    return analogsignal, spike_times, sampling_frequency, time

### FIRING RATE ###
def get_firing_rate(spike_times, recording, sampling_rate):
    num_spikes = len(spike_times)
    fr = num_spikes / (len(recording) / sampling_rate)
    
    return fr

def calculate_instantaneous_firing_rate(spike_times, recording, sampling_rate, step_size, window):
    time_bins = np.arange(0, len(recording)/sampling_rate, step_size)
    
    ifr_ls = [len(np.nonzero((spike_times > bin) & (spike_times < bin + window))[0]) / window for bin in time_bins]
    
    return ifr_ls, time_bins

# TODO: CV skew kurtosis etc. features in ISI histogram probably can be applied to IFR too

### WAVEFORM FEATURES ###
def get_mean_amplitude(spike_times, recording, sampling_rate, window):
    """ amplitude defined as peak-trough change """
    amplitude_ls = []
    for spike in range(len(spike_times)):
        if spike_times[spike]*sampling_rate - window < 0:
            spike_segment = recording[0:int(spike_times[spike]*sampling_rate+window)]
            print('start spike')
        elif spike_times[spike]*sampling_rate + window > len(recording):
            spike_segment = recording[int(spike_times[spike]*sampling_rate-window):int(len(recording))]
            print('end spike')
        else:  
            spike_segment = recording[int(spike_times[spike]*sampling_rate-window):int(spike_times[spike]*sampling_rate+window)]
        
        # print(spike_segment)
        
        amplitude = np.max(spike_segment) - np.min(spike_segment)
        amplitude_ls.append(amplitude)
        # print(amplitude)
    mean_amplitude = np.mean(amplitude_ls)
    
    return mean_amplitude, amplitude_ls

def get_mean_amplitude2(spike_times, recording, sampling_rate, window):
    """ amplitude defined as larger of the peak or trough - baseline as estimated via mean of spike segment"""
    amplitude_ls = []
    for spike in range(len(spike_times)):
        if spike_times[spike]*sampling_rate - window < 0:
            spike_segment = recording[0:int(spike_times[spike]*sampling_rate+window)]
            print('start spike')
        elif spike_times[spike]*sampling_rate + window > len(recording):
            spike_segment = recording[int(spike_times[spike]*sampling_rate-window):int(len(recording))]
            print('end spike')
        else:  
            spike_segment = recording[int(spike_times[spike]*sampling_rate-window):int(spike_times[spike]*sampling_rate+window)]
        
        baseline_mean = np.mean(spike_segment)
        
        amplitude = np.max(np.abs(spike_segment)) - baseline_mean
        amplitude_ls.append(amplitude)
        # print(amplitude)
    mean_amplitude = np.mean(amplitude_ls)
    
    return mean_amplitude, amplitude_ls

### INTERSPIKE INTERVAL ###
def get_ISI_metrics(spike_times):
    """ ISI mode not used as function not built to handle multiple modes """
    # ISI = [spike_times[i] - spike_times[i-1] for i in range(1,len(spike_times))]
    ISI = np.diff(spike_times)
    ISI_cv = np.std(ISI) / np.mean(ISI)
    ISI_skew = stats.skew(ISI)
    ISI_kurtosis = stats.kurtosis(ISI)
    ISI_mean = np.mean(ISI)
    isi_binned, counts = np.unique([round(isi,3) for isi in ISI], return_counts=True) # rounding to millisecond
    ISI_mode = isi_binned[np.argwhere(counts == np.amax(counts))] # may be binomial or multinomial, not used
    
    return ISI_cv, ISI_skew, ISI_kurtosis, ISI_mean, ISI_mode

### BURSTING ###
def burst_detection_neuroexplorer(spike_times, 
                                  recording, 
                                  sampling_rate, 
                                  min_surprise = 5, 
                                  min_numspikes = 3):
    
    """ 
    bursting detection via Poisson surprise
    adapted from neuroexplorer method based on description in manual
    https://plexon.com/wp-content/uploads/2017/06/NeuroExplorer-v5-Manual.pdf page 133
    
    """
    
    burst_dict = {'burst_start_spike' : [],
              'burst_end_spike' : [],
              'burst_numspikes' : [],
              'burst_surprise' : []}

    mean_firing_rate = get_firing_rate(spike_times, recording, sampling_rate)
    mean_ISI = np.mean(np.diff(spike_times))
    # median_ISI = np.median(np.diff(spike_times))

    ISI_to_start_burst = mean_ISI / 2
    ISI_to_end_burst = mean_ISI

    ISI = np.diff(spike_times)
    j=0

    while j+1 < len(ISI):
        isi = ISI[j]
        surprise_list = []
        # print('ISI j', j)
        
        best_surprise = 0
        best_numspikes = 0
        burst_start_spike = 0
        burst_end_spike = 0
        
        if isi < ISI_to_start_burst and ISI[j+1] < ISI_to_start_burst:
            num_spikes = 3 # always starts with minimum 3 spikes
            surprise = - np.log10(np.exp(-mean_firing_rate) * np.power(mean_firing_rate, num_spikes) / np.math.factorial(num_spikes))
            surprise_list.append(surprise)
            best_surprise = surprise
            best_numspikes = num_spikes
            burst_start_spike = j
            burst_end_spike = j + num_spikes
            # add spike
            # if j + num_spikes -1 < len(ISI) == True:
            #     while ISI[j+num_spikes-1] < ISI_to_end_burst:
            while j + num_spikes - 1 < len(ISI) - 1 and ISI[j+num_spikes-1] < ISI_to_end_burst:
                    num_spikes = num_spikes + 1
                    surprise = - np.log10(np.exp(-mean_firing_rate) * np.power(mean_firing_rate, num_spikes) / np.math.factorial(num_spikes))
                    surprise_list.append(surprise)

                    if surprise >= best_surprise:
                        best_surprise = surprise
                        best_numspikes = num_spikes
                        # burst_start_spike = j # doesnt change until backward
                        burst_end_spike = j + best_numspikes
                
            # backward bursts
            backward_surprise_list = []
            for i in range(num_spikes,3,-1): # Start -1 because of spike to ISI conversion
                backward_surprise = - np.log10(np.exp(-mean_firing_rate) * np.power(mean_firing_rate, i) / np.math.factorial(i))
                backward_surprise_list.append(backward_surprise)
                if backward_surprise >= best_surprise:
                    best_surprise = backward_surprise
                    best_numspikes = i
                    # burst_end_spike = j + best_numspikes # doesnt change in backward
                    burst_start_spike = burst_end_spike - i
                
        if best_numspikes >= min_numspikes and best_surprise > min_surprise:
            # print('burst detected!')
            burst_dict['burst_start_spike'].append(burst_start_spike)
            burst_dict['burst_end_spike'].append(burst_end_spike)
            burst_dict['burst_numspikes'].append(best_numspikes)
            burst_dict['burst_surprise'].append(best_surprise)
            j = j + best_numspikes - 1
        else:
            j = j + 1
        
    return burst_dict

### SYNCHRONY ###
def get_synchrony_features(spike_times, 
                           time_bin_size = 0.01, 
                           max_lag_time = 0.5, 
                           avg_window_size = 5,
                           significance_level = 0.05, 
                           to_plot = False):
    
    """ method from 'Burst and oscillation as disparate neuronal properties' Kaneoke & Vitek (1996) """
    
    # Create bins
    bins = np.arange(0, max_lag_time + time_bin_size, time_bin_size)

    # Initialize autocorrelogram
    autocorrelogram = np.zeros_like(bins[:-1], dtype=float)

    # Calculate autocorrelogram
    for i in range(len(bins) - 1):
        time_lag = bins[i]
        spike_pairs = spike_times[(spike_times >= time_lag) & (spike_times < time_lag + max_lag_time)]
        autocorrelogram[i] = len(spike_pairs)

    # Normalize autocorrelogram
    autocorrelogram /= np.sum(autocorrelogram)
    moving_avg_window = np.ones(avg_window_size) / avg_window_size
    smoothed_autocorrelogram = signal.convolve(autocorrelogram, moving_avg_window, mode='same')
    
    w = np.linspace(2*2*np.pi, 50*2*np.pi, 100)
    pgram = signal.lombscargle(bins[:-1], smoothed_autocorrelogram, w, normalize=True)
    
    # Calculate the threshold based on the significance level
    threshold = np.percentile(pgram, (1 - significance_level) * 100)
    
    peaks, peak_props = signal.find_peaks(pgram, height=threshold)
    freq_peaks = w[peaks] / (2*np.pi)
    max_peak_freq = w[peaks[peak_props['peak_heights'].argmax()]] / (2* np.pi)

    frequency_bands = {
        'delta': (0.1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50),
    }

    peak_bands = []
    for band, (lower, upper) in frequency_bands.items():
        peak_bands.extend([band for p in freq_peaks if lower <= p < upper])
    
    if to_plot == True:
        fig,axes = plt.subplots(3,1)
        axes[0].plot(bins[:-1], autocorrelogram)
        axes[0].set_title('Autocorrelogram')
        axes[0].set_xlabel('Time Lag (ms)')
        axes[0].set_ylabel('Normalized \n Spike Counts')
        
        axes[1].plot(bins[:-1], smoothed_autocorrelogram)
        axes[1].set_xlabel('Time Lag (ms)')
        axes[1].set_ylabel('Normalized \n Spike Counts')
        axes[1].set_title('Smoothed autocorrelogram')
        
        axes[2].plot(w/(2*np.pi), pgram)
        axes[2].axhline(y=threshold, color='r', label='p = ' + str(significance_level))
        axes[2].set_ylabel('Magnitude')
        axes[2].set_xlabel('Frequency (Hz)')
        
        fig.legend()
        fig.tight_layout()
    
    return max_peak_freq, freq_peaks, peak_bands