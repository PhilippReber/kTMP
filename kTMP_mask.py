import numpy as np
import sounddevice as sd
from numpy.random import permutation
from scipy.io import savemat
from scipy.io import wavfile
import matplotlib.pyplot as plt
import os.path as op
import scipy.io.wavfile as wavfile
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.stats import zscore


class kTMP_mask:
    """
    Example 1
    ---------
    from kTMP_mask import kTMP_mask

    Mask = kTMP_mask()
    modulators = [4, (17,28)]  # should be a list. if element of list is a tuple, modulation will blend randomly across.
    duration = 10

    Mask.make_mask(modulators=modulators, duration=duration)
    Mask.save(file_name='mask_20min')
    Mask.plot()
    Mask.play()

    Example 2
    ---------
    from kTMP_mask import kTMP_mask
    
    Mask = kTMP_mask()
    modulators = None
    duration = 60
    sampling_rate = 44100

    peak_freqs, peak_powers_n = Mask.analyze_wav('/path/to/file.wav')  # load in wav file to reproduce its spectrum 
    Mask.make_mask(modulators=modulators, duration=duration, carriers=peak_freqs, carrier_amps=peak_powers_n, sampling_rate=sampling_rate)
    
    """

    def __init__(self):
        self.signal = 0
        self.sampling_rate = 0
        self.duration = 0
        
    def phase_continuous_signal(self, start_freq, end_freq, duration, sampling_rate):
        times = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
        frequencies = np.linspace(start_freq, end_freq, len(times))
        phase = 2 * np.pi * np.cumsum(frequencies) / sampling_rate
        signal = np.sin(phase)
        
        return signal

    def make_modulator(self, fmin, fmax, sampling_rate, duration, transition_duration):
        freqs = np.arange(fmin, fmax, 1)
        modulator = np.array([])
        n_times = int(duration * sampling_rate)
        while len(modulator)/sampling_rate < duration:
            freqs = permutation(freqs)
            for ix in range(len(freqs) - 1):
                start_freq = freqs[ix]
                end_freq = freqs[ix + 1]
                transition_signal = self.phase_continuous_signal(start_freq, end_freq, transition_duration, sampling_rate)
                if len(modulator) > 0:
                    if np.diff(modulator)[-1] < 0:
                        transition_signal *= -1
                modulator = np.append(modulator, transition_signal)
        
        return modulator[:n_times]
    
    def make_mask(self, modulators, duration, carriers=[3500, 7000, 10500, 14000], 
                  carrier_amps=[1, 1, 2, 1], sampling_rate=36000, transition_duration=1,
                  intercept=2):
        """
        Parameters
        ----------
        modulators : List of integers or tuples | None
            List of frequencies that will be applied as amplitude modulation to the kHz carrier. 
            If elements are tuples (fmin,fmax), the modulation will blend randomly across frequency in that range.
            If None, no modulation will be applied.
        duration : int
            The total duration of the mask in seconds.
        carriers : List of integers
            The carrier of the mask.
        carrier_amps : list of integers
            The individual volume of each carrier.
        sampling_rate : int
            The sampling rate of the returned mask.
        transition_duration : int
            How long each single frequency should last, if modulators are blended across.
        intercept : int
            The intercept of the modulation 
        """
        
        if not isinstance(modulators, (list, type(np.array))) and modulators is not None:
            raise Exception('modulators argument has to be list or array')
        if len(carriers) != len(carrier_amps):
            raise Exception('carriers and carrier_amps must be of equal length')
        
        all_times = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)
        signal = np.sum([amp * np.sin(2 * np.pi * carrier_freq * all_times) for carrier_freq, amp in zip(carriers, carrier_amps)], axis=0)

        if modulators:
            for f in modulators:
                if isinstance(f, tuple):
                    fmin, fmax = f
                    modulator = self.make_modulator(fmin, fmax, sampling_rate, duration, transition_duration)
                elif isinstance(f, int):
                    modulator = np.sin(2 * np.pi * f * all_times)
            
                signal *= 0.5 * (intercept + modulator)

        self.signal = signal
        self.sampling_rate = sampling_rate
        self.duration = duration

    def plot(self):
        # plot the signal
        plt.plot(self.signal)
        plt.show()
        
    def play(self):
        # play the signal
        sd.play(self.signal, samplerate=self.sampling_rate)
        sd.wait()

    def save(self, file_name):
        # save to matlab and wav format
        path = op.join(op.dirname(__file__), file_name)
        mdict = {'signal': self.signal, 'sampling_rate': self.sampling_rate, 'duration': self.duration}
        savemat(f'{path}.mat', mdict)
        wavfile.write(f'{path}.wav', self.sampling_rate, self.signal)

    def analyze_wav(self, path, plot=False):
        """
        Read in a wav file to reproduce specific spectral profiles.

        Parameters
        ----------
        path : str 
            The file path to the wav file.

        Returns
        -------
        peak_freqs : array-like
            The frequency bins of the FFT that have extreme power.
        peak_powers_n : array-like
            The normalized power values associated with each frequency bin.
        """
        # read the wav file
        sfreq, data = wavfile.read(path)
        if len(data.shape) > 1:
            data = data[:,0]
        
        # compute fft
        power = np.abs(fft(data))
        freq_bins = np.abs(fftfreq(len(data), d=1/sfreq))

        # extract spectral peaks to reproduce
        freq_sfreq = len(freq_bins[(freq_bins > 0) & (freq_bins <= 1)])
        all_peak_ixs = find_peaks(power, distance=20*freq_sfreq)[0]
        z_power = zscore(power)
        peak_ixs = [ix for ix in all_peak_ixs if z_power[ix] > 1.641 and freq_bins[ix] > 1000]
        peak_freqs = freq_bins[peak_ixs]
        peak_powers = power[peak_ixs]
        peak_powers_n = (peak_powers - np.min(peak_powers)) / (np.max(peak_powers) - np.min(peak_powers))      

        if plot:
            plt.figure()
            for ix in peak_ixs:
                plt.scatter(freq_bins[ix], power[ix], color='red', s=8)
            plt.plot(freq_bins, power)
            plt.xlim(0,16000)
            plt.show()

        return peak_freqs, peak_powers_n


def main():
    Mask = kTMP_mask()
    modulators = [(2,4)]
    duration = 60
    sampling_rate = 44100

    peak_freqs, peak_powers_n = Mask.analyze_wav('/path/to/file.wav')
    Mask.make_mask(modulators=modulators, duration=duration, carriers=peak_freqs, carrier_amps=peak_powers_n, sampling_rate=sampling_rate)
    Mask.save('mask')
    Mask.plot()
    Mask.play()

if __name__ == '__main__':
    main()
