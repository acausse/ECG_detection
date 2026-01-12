import numpy as np
from scipy import stats

def get_timeWinsTemplatedSignal(signal, timeWins, nTimePoints = None, filling = np.nan):
    """
    return templatedSignal
    
    signal must be a 'pure' np.array, VECTOR shape. If it is an array of lists of len 1, please use: signal = np.ravel(signal) before passing signal into the function 
    timeWins is a np.array of shape (nWins, 2) / 2 for start and for stop
    """
    if nTimePoints is None:
        nTimePoints = signal.shape[0]
    templatedSignal = np.full(nTimePoints, filling)
    for timeWin in timeWins:
        templatedSignal[timeWin[0]:timeWin[1]] = signal[timeWin[0]:timeWin[1]]
    return templatedSignal

def get_timeWinsIntersect(itwA, itwB, lastPoint, firstPoint=0, ifDisplay=False, lfp=None):
    """
    Computes the complementary (non-overlapping) time windows with respect to two input sets of time intervals.

    return itwTot

    Parameters
    ----------
    itwA : np.ndarray
        First set of time windows (Nx2 array, each row is [start, end]).
    itwB : np.ndarray
        Second set of time windows (Nx2 array, each row is [start, end]).
    lastPoint : int
        The maximum time point of the signal (exclusive).
    firstPoint : int, optional
        The minimum time point of the signal (inclusive), default is 0.
    ifDisplay : bool, optional
        If True, check and print whether the output windows overlap with input intervals using the `lfp` signal.
    lfp : np.ndarray, optional (use None if ifDisplay=False)
        Local field potential signal to use for overlap checking when `ifDisplay` is True.

    Returns
    -------
    itwTot : np.ndarray
        Array of non-overlapping time windows relative to the union of `itwA` and `itwB`.
        These intervals represent all time periods not covered by either `itwA` or `itwB`.

    Notes
    -----
    - If either `itwA` or `itwB` is empty, the result will be the other interval set.
    - This function uses a binary mask (`skull`) to mark points not covered by any input interval.
    - The resulting intervals are created based on gaps in this mask.
    - If `ifDisplay` is True, the function checks whether the resulting windows fully cover the regions not spanned by `itwA` and `itwB`.
    - A warning is printed if `firstPoint` lies within an excluded region, potentially leading to a misalignment of the first output window.
    """
    skull=np.ones(lastPoint, bool)
    
    if np.logical_or(itwA.shape[0]==0, itwA.shape[1]==0):
        itwTot=itwB
    elif np.logical_or(itwB.shape[0]==0, itwB.shape[1]==0):
        itwTot=itwA
    else:
        for itw in itwA:
            skull[itw[0]:itw[1]]=False
        for itw in itwB:
            skull[itw[0]:itw[1]]=False

        edgeInds=np.where(np.diff(np.where(skull)[0])>1)[0]
        trues=np.where(skull)[0]

        itwTot=[[trues[edgeInds][i]+1, trues[edgeInds+1][i]] for i in range(edgeInds.shape[0])]

        if skull[0]==False: # starts by False (not detected window)
            itwTot.insert(0, [firstPoint, trues[0]])

        if skull[-1]==False:
            itwTot.append([trues[-1], lastPoint])

        itwTot=np.array(itwTot)

        if ifDisplay:
            lfpTpItwA=get_timeWinsTemplatedSignal(lfp, itwA)
            lfpTpItwB=get_timeWinsTemplatedSignal(lfp, itwB)
            lfpTpItwTot=get_timeWinsTemplatedSignal(lfp, itwTot)

            if np.sum(np.isnan(lfpTpItwA)==False)+np.sum(np.isnan(lfpTpItwB)==False) == np.sum(np.isnan(lfpTpItwTot)==False):
                print('NON-overlapping windows')
            else:
                print('Overlapping windows')

        if itwTot[0][0]>itwTot[0][1]:
            print('WARNING: firstPoint is greater than the first time window. Consider editing firstPoint.')
    return itwTot


def get_timeWins4ContiguousPoints(array):
    """
    Identifies contiguous blocks in a 1D array and returns the start and end indices of each block along with the corresponding values.

    Parameters:
        array (numpy.ndarray): Input array of shape (N,) with integer values.

    Returns:
    return winTimes, winValue
        winTimes (numpy.ndarray): Array of shape (x, 2), where each row contains the start and end indices of a contiguous block.
        winValue (numpy.ndarray): Array of shape (x,) containing the values of the contiguous blocks.
    """
    # Identify the indices where the value changes
    change_indices = np.where(np.diff(array) != 0)[0] + 1
    
    # Include the start and end of the array
    start_indices = np.insert(change_indices, 0, 0)
    end_indices = np.append(change_indices, len(array))
    
    # Generate winTimes as start and end index pairs
    winTimes = np.column_stack((start_indices, end_indices))
    
    # Generate winValue as the values of the blocks
    winValue = array[start_indices]
    
    return winTimes, winValue


def filt_butter(data, lowcut=0, highcut=50, btype='highpass', sr=1250.0, order = 5, CAUSAL=True):
    """
    return data_filt
    For a highpass filter, define highcut
    """
    from scipy.signal import butter, lfilter, filtfilt
    
    def butter_(lowcut, highcut,btype, sr, order=order):
        nyq = 0.5 * sr
        low = lowcut / nyq
        high = highcut / nyq
        if btype=='highpass':
            b, a = butter(order, high, btype=btype)
        elif btype=='lowpass':
            b, a = butter(order, low, btype=btype)
        elif btype=='bandpass':
            b, a = butter(order, [low, high], btype=btype)
        return b, a
    
    b, a = butter_(lowcut, highcut,btype, sr, order=order)
    if CAUSAL:
        data_filt = lfilter(b, a, data)
    else:
        data_filt = filtfilt(b, a, data)
    return data_filt

def detect_ecg_R_peaks(ecg_signal, sr,thresh_agreement=3, thresh_std=2, cut_hz=0.5,
                       extend4Max_ms=100, 
                       pan_tomkins=True,hamilton=True,christov=True,
                       stationary_wavelet=True,two_average=True, engzee=False):
    """
    Detect R-peaks in an ECG signal using multi-algorithm agreement.

    This function applies several established ECG R-peak detection algorithms
    (via the `py-ecg-detectors` package) and combines their outputs using an
    agreement-based strategy. Detected peaks are first aligned to local extrema,
    thresholded based on signal statistics, and finally retained if detected
    by a minimum number of algorithms.

    Parameters
    ----------
    ecg_signal : np.ndarray
        One-dimensional ECG signal.
    sr : float
        Sampling rate of the ECG signal in Hz.
    thresh_agreement : int, optional
        Minimum number of detectors that must agree for an R-peak to be accepted.
        Default is 3.
    thresh_std : float, optional
        Threshold in units of signal standard deviation used to reject low-amplitude
        detections. Default is 2.
    cut_hz : float, optional
        High-pass cutoff frequency (Hz) applied to remove slow baseline components.
        Default is 0.5 Hz.
    extend4Max_ms : float, optional
        Time window (in milliseconds) used to refine R-peak locations by aligning
        detections to local extrema. Default is 100 ms.
    pan_tomkins : bool, optional
        Enable Pan–Tompkins detector.
    hamilton : bool, optional
        Enable Hamilton detector.
    christov : bool, optional
        Enable Christov detector.
    stationary_wavelet : bool, optional
        Enable stationary wavelet transform (SWT) detector.
    two_average : bool, optional
        Enable two-average detector.
    engzee : bool, optional
        Enable Engzee detector.

    Returns
    -------
    r_peaks : dict
        Dictionary containing intermediate and final R-peak detections:
        - 'first' : raw detections from each detector
        - 'aligned' : detections aligned to local signal extrema
        - 'above' : detections passing amplitude thresholding
        - 'agree' : final R-peaks detected by at least `thresh_agreement` algorithms

        The final R-peak indices are available via:
        ```python
        r_peaks['agree']
        ```

    Notes
    -----
    Increasing `thresh_agreement` improves specificity but may reduce sensitivity.
    """
    
    from ecgdetectors import Detectors
    
    # Initialize
    detectors = Detectors(sr)
    # High pass at 0.5Hz to remove slow components (movement)
    signal=filt_butter(ecg_signal, highcut=cut_hz, btype='highpass')
    
    r_peaks={}
    r_peaks['first']={}

    if pan_tomkins:
        r_peaks['first']['pan'] = detectors.pan_tompkins_detector(signal)
    if hamilton:
        r_peaks['first']['ham'] = detectors.hamilton_detector(signal)
    if christov:
        r_peaks['first']['chr'] = detectors.christov_detector(signal)
    if stationary_wavelet:
        r_peaks['first']['swt'] = detectors.swt_detector(signal)
    if two_average:
        r_peaks['first']['twa'] = detectors.two_average_detector(signal)
    if engzee:
        r_peaks['first']['eng'] = detectors.engzee_detector(signal)

    # Align detected R-peaks to local maximum
    extend4Max_sr=int(extend4Max_ms/1000 * sr)

    r_peaks['aligned']={}
    for key in r_peaks['first'].keys():
        rp=np.array(r_peaks['first'][key])
        for rpi in range(len(rp)):
            MAX=np.max(signal[rp[rpi]-extend4Max_sr : rp[rpi]+extend4Max_sr])
            MIN=np.min(signal[rp[rpi]-extend4Max_sr : rp[rpi]+extend4Max_sr])
            if abs(MAX)>abs(MIN):
                ind=np.argmax(signal[rp[rpi]-extend4Max_sr : rp[rpi]+extend4Max_sr])
            else:
                ind=np.argmin(signal[rp[rpi]-extend4Max_sr : rp[rpi]+extend4Max_sr])
            rp[rpi]=rp[rpi] + (ind-extend4Max_sr)
        r_peaks['aligned'][key]=np.array(rp)

    # Condition on threshold
    m=np.mean(signal)
    std=np.std(signal)

    r_peaks['above']={}
    for key in r_peaks['first'].keys():
        rp=np.array(r_peaks['aligned'][key])

        r_peaks['above'][key]=  rp[ np.logical_or(signal[rp] > m+ thresh_std*std,
                                     signal[rp] < m- thresh_std*std ) ]

    # Agreement
    rp_unique=np.array(np.unique(np.concatenate(list(r_peaks['above'].values()))), int)
    r_peaks['agree_arrays']=np.array([[rp_un in value for value in r_peaks['above'].values()]
                           for rp_un in rp_unique])
    r_peaks['agree_vals']= r_peaks['agree_arrays'].sum(1)
    r_peaks['agree']=rp_unique[r_peaks['agree_vals']>=thresh_agreement]
    return r_peaks


def get_ecg_phases(ecg, r_peaks):
    """
    Compute a continuous cardiac phase signal from detected R-peaks.

    This function assigns a phase ranging from 0 to 2π to each sample
    of an ECG signal, with phase resets occurring at each R-peak.
    Phases are linearly interpolated between consecutive R-peaks,
    enabling phase-based and phase-locked analyses of cardiac activity.

    Parameters
    ----------
    ecg : np.ndarray
        One-dimensional ECG signal.
    r_peaks : np.ndarray
        Array of R-peak sample indices (e.g., output of `detect_ecg_R_peaks['agree']`).

    Returns
    -------
    ecg_phases : np.ndarray
        One-dimensional array of the same length as `ecg`, containing
        the cardiac phase (in radians) at each time point, spanning
        from 0 to 2π between successive R-peaks.
    """
    phases_start=np.linspace(0, np.pi * 2, r_peaks[0])
    phases_body=np.concatenate([np.linspace(0, np.pi * 2, r_peaks[trigi+1]-r_peaks[trigi]) for trigi in range(len(r_peaks)-1)])
    phases_end=np.linspace(0, np.pi * 2, ecg.shape[0]-r_peaks[-1])

    ecg_phases=np.concatenate([phases_start, phases_body, phases_end])
    return ecg_phases


def get_ecg_maxPol(polarities, polTriggers, totalTriggers):
    """
    return maxPol
    """
    maxPolInd=np.argmax([polTriggers[polarity].shape[0] / totalTriggers.shape[0] for polarity in polarities])
    maxPol=polarities[maxPolInd]
    return maxPol

def get_ecg_hrv(r_peaks, sr, percThresh=0.3):
    """
    return hrv_ms
    """
    hrv_ms=(np.diff(r_peaks)/sr)*1000
    hrv_ms_filt=hrv_ms[np.logical_and(hrv_ms>=np.median(hrv_ms)-np.median(hrv_ms)*percThresh,
                                      hrv_ms<np.median(hrv_ms)+np.median(hrv_ms)*percThresh)]
    return hrv_ms_filt
    

def get_ecg_rmssd(rr_intervals):
    """
    Calculate RMSSD (Root Mean Square of Successive Differences) from RR intervals.
    Expect values aroun 20-150ms
    
    Parameters:
    rr_intervals (list or numpy array): Sequence of RR intervals in milliseconds or seconds.

    Returns:
    float: RMSSD value.
    """
    # Ensure input is a numpy array
    rr_intervals = np.array(rr_intervals)

    # Calculate successive differences of RR intervals
    diff_rr = np.diff(rr_intervals)

    # Square the differences
    squared_diff = diff_rr ** 2

    # Calculate the mean of the squared differences
    mean_squared_diff = np.mean(squared_diff)

    # Compute the root mean square of the differences
    rmssd = np.sqrt(mean_squared_diff)

    return rmssd


def detect_ecg_mvt_artefacts(ecg, sr, highcut_hz=50,  thresh_z=2, ext_ms=10, thresh_dur_ms=200):
    """
    Detects movement artifacts in an ECG signal.
    
    VERY CRAPPY  - extremely sensitive to noise - not recommended to use this
    
    Parameters:
    - ecg (array): The ECG signal to process.
    - sr (int): Sampling rate of the ECG signal (in Hz).
    - thresh_z (float): Threshold for z-score to detect significant deviations. Default is 2.
    - ext_ms (float): Time (in milliseconds) to extend detected artifact regions on both sides. Default is 10 ms.
    - thresh_dur_ms (float): Minimum duration (in milliseconds) for artifact events to be retained. Default is 200 ms.

    Returns:
    - detectedWins (array): Array of artifact time windows [start, end] in samples.
    - detectStream (array): Binary array indicating artifact presence (1) or absence (0) at each sample.
    """
    # Step 1: High-pass filter the ECG signal to remove low-frequency components.
    trace_f = filt_butter(ecg, highcut=highcut_hz, btype='highpass')

    # Step 2: Compute the envelope of the z-scored derivative of the filtered signal.
    env = np.abs(stats.zscore(np.diff(trace_f)))

    # Step 3: Detect contiguous time windows where the envelope exceeds the z-score threshold.
    winTimes, winValue = get_timeWins4ContiguousPoints(np.array(env > thresh_z, int))

    # Step 4: Extend detected windows by a user-defined margin (ext_ms) on both sides.
    lastPoint = ecg.shape[0]  # Total number of samples in the ECG signal.
    tws = np.column_stack([
        winTimes[winValue == 1][:, 0] - int((ext_ms / 1000 * sr)),  # Extend start points.
        winTimes[winValue == 1][:, 1] + int((ext_ms / 1000 * sr))   # Extend end points.
    ])

    # Step 5: Merge overlapping or intersecting time windows.
    tws_ext = get_timeWinsIntersect(tws, tws, lastPoint)

    # Step 6: Clip extended windows to ensure they stay within valid signal boundaries.
    tws_ext[:, 0][tws_ext[:, 0] <= 0] = 0  # Prevent start indices from being negative.
    tws_ext[:, 1][tws_ext[:, 1] > lastPoint] = lastPoint  # Prevent end indices from exceeding signal length.

    # Step 7: Calculate the duration of each detected window in milliseconds.
    tws_dur_ms = np.diff(tws_ext)[:, 0] / sr * 1000

    # Step 8: Retain only windows that exceed the minimum duration threshold (thresh_dur_ms).
    detectedWins = tws_ext[tws_dur_ms > thresh_dur_ms]

    # Step 9: Generate a binary detection stream from the detected artifact windows.
    skull = np.ones(lastPoint)  # Create a placeholder array with all ones.
    detectStream = get_timeWinsTemplatedSignal(skull, detectedWins, filling=0)  # Apply detected windows.

    # Step 10: Return the detected artifact windows and the binary detection stream.
    return detectedWins, detectStream
