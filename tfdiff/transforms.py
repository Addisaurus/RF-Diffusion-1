import numpy as np
import scipy.signal as spsig

def resample_a_to_b(a, a_fs, b_fs, **kwargs):
    """Resample signal a with sample rate a_fs to sample rate b_fs.
    See https://docs.scipy.org/doc/scipy-1.1.0/reference/generated/scipy.signal.resample_poly.html for additional
    keyword arguments: axis=0, and window=('kaiser', 5.0)
    """
    if a_fs == b_fs:
        # No need to resample
        return a

    # scipy.signal provides several methods to resample a signal.
    #
    # Simplest (possibly slow) approach is the built-in polyphase

    # Check for non-integer sampling rate (e.g. vsg-apco25 is 195312.5 samples/sec)
    if np.mod(a_fs, 1) != 0:
        assert np.mod(2*a_fs, 1) == 0, "Non-integer sample rate cannot be fixed by multiplying down and up by 2.  Implement something smarter."
        a_fs *= 2
        b_fs *= 2

    # SSCLANT_Signal_Combining_20180627.docx suggests that if the polyphase approach
    # is too slow, then "the recommended technique for combining SOI recordings with
    # the background recordings is to use a FIR-compensated CIC filter to increase
    # the sample rate of the SOI within a factor of 2 of the sample rate of the
    # background. Then, use a Farrow filter to do the final rate matching. Finally,
    # multiply the upsampled SOI by a complex exponential to shift its center
    # frequency to the desired location in the background."
    # TODO: Implement the SSCLANT recommendation above.
    return spsig.resample_poly(a, b_fs, a_fs, **kwargs)

def lowpass_filter(signal, freq_lo, freq_hi, fs, num_taps=512, window='hamming'):
    # Construct a freq-symmetric LP filter with cutoff freq equal to half the bw of the desired signal
    cutoff = (freq_hi - freq_lo) / 2.0 - 1.0  # do half bandwidth b/c firwin produces a freq-symmetric LP filter
    h_lowpass = spsig.firwin(num_taps, cutoff, window=window, pass_zero=True, fs=fs)
    return spsig.convolve(signal, h_lowpass, mode='same')

def bandpass_filter(signal, freq_lo, freq_hi, fc, fs, num_taps=512, window='hamming'):
    # Construct a freq-symmetric LP filter with cutoff freq equal to half the bw of the desired signal
    fc_baseband = (freq_lo + freq_hi) / 2.0 - fc
    bw = freq_hi - freq_lo
    cutoff = bw / 2.0 - 1.0  # do half bandwidth b/c firwin produces a freq-symmetric LP filter
    # Do -1.0 since cutoff must be "less" than fs / 2
    h_lowpass = spsig.firwin(num_taps, cutoff, window=window, pass_zero=True, fs=fs)
    # 2) Frequency shift the LP filter to center it over the desired frequency band
    h_bandpass = h_lowpass * np.exp((2.0j * np.pi * fc_baseband / fs) * np.arange(0, num_taps))
    return spsig.convolve(signal, h_bandpass, mode='same')

def shift_frequency(signal, fs, delta_f):
    dt = 1.0 / fs
    return signal * np.exp(2.0j * np.pi * dt * delta_f * np.arange(len(signal)))

def shift_and_lowpass(signal, freq_lo, freq_hi, fc, fs, num_taps=512, window='hamming'):
    delta_f = fc - 0.5 * float(freq_hi + freq_lo)
    dt = 1.0 / fs
    shifted_signal = signal * np.exp(2.0j * np.pi * dt * delta_f * np.arange(len(signal)))
    # Construct a freq-symmetric LP filter with cutoff freq equal to half the bw of the desired signal
    cutoff = (freq_hi - freq_lo) / 2.0 - 1.0  # do half bandwidth b/c firwin produces a freq-symmetric LP filter
    h_lowpass = spsig.firwin(num_taps, cutoff, window=window, pass_zero=True, fs=fs)
    return spsig.convolve(shifted_signal, h_lowpass, mode='same')

def iq_center_norm_transform(raw_data, meta, sample_rate, startind=0, stopind=None):
    # Baseband shift and lowpass filter the signal
    fs = meta[2]
    signal = shift_and_lowpass(raw_data, meta[3], meta[4], meta[1], fs)
    signal = resample_a_to_b(signal, fs, sample_rate)

    # Complex normalize (divide by max magnitude complex value)
    raw_mean = np.mean(signal, axis=0)
    centered = signal - raw_mean
    max_val = centered[np.argmax(np.absolute(centered))]

    # Only return desired subset of signal
    if stopind is not None and stopind > startind:
        subset = centered[startind:stopind] / max_val
    else:
        subset = centered[startind:] / max_val

    # Transform to 2 dimensional, with I and Q
    full_data = np.zeros((subset.size, 2))
    full_data[:,0] = np.real(subset)
    full_data[:,1] = np.imag(subset)
    return full_data

def iq_center_norm_resample(raw_data, resample_rate=1, startind=0, stopind=-1):
    signal = resample_a_to_b(raw_data, 1, resample_rate)

    # Complex normalize (divide by max magnitude complex value)
    raw_mean = np.mean(signal, axis=0)
    centered = signal - raw_mean
    max_val = centered[np.argmax(np.absolute(centered))]

    # Only return desired subset of signal
    if stopind is not None and stopind > startind:
        subset = centered[startind:stopind] / max_val
    else:
        subset = centered[startind:] / max_val

    # Transform to 2 dimensional, with I and Q
    full_data = np.zeros((subset.size, 2))
    full_data[:,0] = np.real(subset)
    full_data[:,1] = np.imag(subset)
    return full_data

def iq_center_norm_resample_preprocessed(raw_data, resample_rate=1, startind=0, stopind=-1):
    signal = resample_a_to_b(raw_data, 1, resample_rate)

    # Complex normalize (divide by max magnitude complex value)
    raw_mean = np.mean(signal, axis=0)
    centered = signal - raw_mean
    max_val = np.max(np.absolute(centered)) #[np.argmax(np.absolute(centered))]

    if stopind is not None and stopind > startind:
        subset = centered[startind:stopind] / max_val
    else:
        subset = centered[startind:] / max_val

    return subset

def resample_only(raw_data, resample_rate=1, startind=0, stopind=-1):
    signal = resample_a_to_b(raw_data, 1, resample_rate)

    # Only return desired subset of signal
    if stopind is not None and stopind > startind:
        subset = signal[startind:stopind]
    else:
        subset = signal[startind:] 

    # Transform to 2 dimensional, with I and Q
    full_data = np.zeros((subset.size, 2))
    full_data[:,0] = np.real(subset)
    full_data[:,1] = np.imag(subset)
    return full_data