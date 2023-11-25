import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neo
import os
from scipy import stats, signal
from WaveformMetrics import *

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

def get_ifr_metrics(ifr):
    ifr_skew = stats.skew(ifr)
    ifr_kurtosis = stats.kurtosis(ifr)
    ifr_mean = np.mean(ifr)
    ifr_var = np.var(ifr)
    fano_factor = ifr_var / ifr_mean
    
    return fano_factor, ifr_mean, ifr_skew, ifr_kurtosis

### WAVEFORM FEATURES ###
def get_mean_amplitude(spike_times, recording, sampling_rate, window):
    """ amplitude defined as peak-trough change """
    amplitude_ls = []
    for spike in range(len(spike_times)):
        if spike_times[spike]*sampling_rate - window < 0:
            spike_segment = recording[0:int(spike_times[spike]*sampling_rate+window)]
            # print('start spike')
        elif spike_times[spike]*sampling_rate + window > len(recording):
            spike_segment = recording[int(spike_times[spike]*sampling_rate-window):int(len(recording))]
            # print('end spike')
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
            # print('start spike')
        elif spike_times[spike]*sampling_rate + window > len(recording):
            spike_segment = recording[int(spike_times[spike]*sampling_rate-window):int(len(recording))]
            # print('end spike')
        else:  
            spike_segment = recording[int(spike_times[spike]*sampling_rate-window):int(spike_times[spike]*sampling_rate+window)]
        
        baseline_mean = np.mean(spike_segment)
        
        amplitude = np.max(np.abs(spike_segment)) - baseline_mean
        amplitude_ls.append(amplitude)
        # print(amplitude)
    mean_amplitude = np.mean(amplitude_ls)
    
    return mean_amplitude, amplitude_ls

def get_ecephys_waveform_metrics(spike_times, recording, window, sampling_rate):
    
    waveforms = np.zeros((len(spike_times), window*2))
    for i, spike in enumerate(spike_times):
        if spike*sampling_rate - window >= 0 and spike*sampling_rate + window <= len(recording):
            waveforms[i,:] = recording[int(spike*sampling_rate)-window:int(spike*sampling_rate)+window]
    
    waveforms = waveforms[~np.all(waveforms == 0, axis=1)]
    mean_waveform = np.mean(waveforms,axis=0)
    
    timestamps = np.linspace(0, window/sampling_rate, window*2)
    
    waveform_duration = calculate_waveform_duration(mean_waveform,timestamps)
    waveform_halfwidth = calculate_waveform_halfwidth(mean_waveform,timestamps)
    waveform_PT_ratio = calculate_waveform_PT_ratio(mean_waveform)
    waveform_TP_time = calculate_waveform_TP_time(mean_waveform, timestamps)
    waveform_positive_spiking = is_positive_spiking(mean_waveform)
    
    # slopes not possible for 12.5kHz sampling rate as insufficient points for linear regression
    # waveform_repolarization_slope = calculate_waveform_repolarization_slope(mean_waveform,timestamps)
    # waveform_recovery_slope = calculate_waveform_recovery_slope(mean_waveform, timestamps)
    
    return waveform_duration, waveform_halfwidth, waveform_PT_ratio, waveform_TP_time, waveform_positive_spiking


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
            burst_end_spike = j + num_spikes - 1
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
                        burst_end_spike = j + best_numspikes - 1
                
            # backward bursts
            backward_surprise_list = []
            for i in range(num_spikes,3,-1): # Start -1 because of spike to ISI conversion
                backward_surprise = - np.log10(np.exp(-mean_firing_rate) * np.power(mean_firing_rate, i) / np.math.factorial(i))
                backward_surprise_list.append(backward_surprise)
                if backward_surprise >= best_surprise:
                    best_surprise = backward_surprise
                    best_numspikes = i
                    # burst_end_spike = j + best_numspikes # doesnt change in backward
                    burst_start_spike = burst_end_spike - i + 1
                
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

def get_burst_metrics(burst_dict, spike_times):
    num_bursts = len(burst_dict['burst_numspikes'])
    mean_surprise = np.mean(burst_dict['burst_surprise'])
    burst_index = np.sum(burst_dict['burst_numspikes']) / len(spike_times) # number of spikes in burst out of all spikes
    
    burst_lengths = [spike_times[burst_dict['burst_end_spike'][i]] - spike_times[burst_dict['burst_start_spike'][i]] for i in range(len(burst_dict['burst_start_spike']))]
    mean_burst_duration = np.mean(burst_lengths)
    var_burst_duration = np.var(burst_lengths)
    
    interburst_lengths = [spike_times[burst_dict['burst_start_spike'][i]] - spike_times[burst_dict['burst_end_spike'][i-1]] for i in range(1,len(burst_dict['burst_start_spike']))]
    mean_interburst_duration = np.mean(interburst_lengths)
    var_interburst_duration = np.var(interburst_lengths)
    
    return num_bursts, mean_surprise, burst_index, mean_burst_duration, var_burst_duration, mean_interburst_duration, var_interburst_duration
    
def get_burst_index(spike_times):
    """ alternative definition of burst index, not using Poisson Surprise method 
        burst index = mean ISI / mode ISI
        use median ISI as proxy for mode ISI due to multimodal distributions
    """
    
    ISI = np.diff(spike_times)
    burst_index = np.mean(ISI) / np.median(ISI)
    
    return burst_index
    
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
    
    if peaks.size > 0:
        freq_peaks = w[peaks] / (2*np.pi)
        max_peak_freq = w[peaks[peak_props['peak_heights'].argmax()]] / (2* np.pi)
        freq_magnitude = peak_props['peak_heights']

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
    
    else:
        freq_peaks = []
        max_peak_freq = np.nan
        peak_bands = []
        freq_magnitude = []
    
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
    
    return max_peak_freq, freq_peaks, peak_bands, freq_magnitude


def autocorrelation(spike_times, bin_size, max_lag):
    bins = np.arange(0, np.max(spike_times) + bin_size, bin_size)
    spike_counts = np.histogram(spike_times, bins=bins)[0]
    autocorr = np.correlate(spike_counts, spike_counts, mode='full')[len(spike_counts) - 1:]
    autocorr_lag = np.arange(0, len(autocorr)) * bin_size
    autocorr = autocorr[autocorr_lag <= max_lag]
    autocorr_lag = autocorr_lag[autocorr_lag <= max_lag]
    return autocorr, autocorr_lag

def get_synchrony_features2(spike_times, 
                           time_bin_size = 0.01, 
                           max_lag_time = 0.5, 
                           significance_level = 0.05, 
                           to_plot = False):
    
    """ method from 'Burst and oscillation as disparate neuronal properties' Kaneoke & Vitek (1996) """

    # autocorrelogram
    autocorr, autocorr_lag = autocorrelation(spike_times, time_bin_size, max_lag_time)
    
    w = np.linspace(2*2*np.pi, 50*2*np.pi, 100)
    pgram = signal.lombscargle(autocorr_lag, autocorr, w, normalize=True)
    
    # Calculate the threshold based on the significance level
    threshold = np.percentile(pgram, (1 - significance_level) * 100)
    
    peaks, peak_props = signal.find_peaks(pgram, height=threshold)
    
    if peaks.size > 0:
        freq_peaks = w[peaks] / (2*np.pi)
        max_peak_freq = w[peaks[peak_props['peak_heights'].argmax()]] / (2* np.pi)
        freq_magnitude = peak_props['peak_heights']

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
    
    else:
        freq_peaks = []
        max_peak_freq = np.nan
        peak_bands = []
        freq_magnitude = []
    
    if to_plot == True:
        fig,axes = plt.subplots(2,1)
        axes[0].plot(autocorr_lag, autocorr)
        axes[0].set_title('Autocorrelogram')
        axes[0].set_xlabel('Time Lag (ms)')
        axes[0].set_ylabel('Normalized \n Spike Counts')
        
        axes[1].plot(w/(2*np.pi), pgram)
        axes[1].axhline(y=threshold, color='r', label='p = ' + str(significance_level))
        axes[1].set_ylabel('Magnitude')
        axes[1].set_xlabel('Frequency (Hz)')
        
        fig.legend()
        fig.tight_layout()
    
    return max_peak_freq, freq_peaks, peak_bands, freq_magnitude