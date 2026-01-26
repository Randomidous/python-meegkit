"""Denoising source separation."""
# Authors:  Nicolas Barascud <nicolas.barascud@gmail.com>
#           Maciej Szul <maciej.szul@isc.cnrs.fr>
#           Roy Eric Wieske <eric.wieske@gmail.com>
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import linalg
from scipy.signal import butter, find_peaks, sosfiltfilt, welch

from .tspca import tsr
from .utils import (
    demean,
    gaussfilt,
    matmul3d,
    mean_over_trials,
    pca,
    smooth,
    theshapeof,
    tscov,
    wpwr,
)


def dss1(X, weights=None, keep1=None, keep2=1e-12):
    """DSS to maximise repeatability across trials.

    Evoked-biased DSS denoising.

    Parameters
    ----------
    X: array, shape=(n_samples, n_chans, n_trials)
        Data to denoise.
    weights: array
        Weights.
    keep1: int
        Number of PCs to retain in function:`dss0` (default=all).
    keep2: float
        Ignore PCs smaller than keep2 in function:`dss0` (default=1e-12).

    Returns
    -------
    todss: array, shape=(n_dss_components, n_chans)
        Denoising matrix to convert X to normalized DSS components.
    from: array, shape=(n_dss_components, n_chans)
        Matrix to convert DSS components back to sensor space.
    pwr0: array
        Power per component (raw).
    pwr1: array
        Power per component (averaged).

    """
    # if demean: # remove weighted mean
    #   X = demean(X, weights)

    # weighted mean over trials (--> bias function for DSS)
    xx, ww = mean_over_trials(X, weights)

    # covariance of raw and biased X
    c0, nc0 = tscov(X, None, weights)
    c1, nc1 = tscov(xx, None, ww)
    c0 /= nc0
    c1 /= nc1

    todss, fromdss, pwr0, pwr1 = dss0(c0, c1, keep1, keep2)

    return todss, fromdss, pwr0, pwr1


def dss0(c0, c1, keep1=None, keep2=1e-9, return_unmixing=True):
    """DSS base function.

    This function allows specifying arbitrary bias functions (as compared to
    the function:`dss1`, which forces the bias to be the mean over trials).

    Parameters
    ----------
    c0: array, shape=(n_chans, n_chans)
        Baseline covariance.
    c1: array, shape=(n_chans, n_chans)
        Biased covariance.
    keep1: int | None
        Number of PCs to retain (default=None, which keeps all).
    keep2: float
        Ignore PCs smaller than keep2 (default=1e-9).
    return_unmixing : bool
        If True (default), return the unmixing matrix.

    Returns
    -------
    todss: array, shape=(n_chans, n_dss_components)
        Matrix to convert X to normalized DSS components.
    fromdss : array, shape=(n_dss_components, n_chans)
        Matrix to transform back to original space. Only returned if
        ``return_unmixing`` is True.
    pwr0: array
        Power per component (baseline).
    pwr1: array
        Power per component (biased).

    Notes
    -----
    The data mean is NOT removed prior to processing.

    """
    # Check size and squareness
    assert c0.shape == c1.shape == (c0.shape[0], c0.shape[0]), \
        "c0 and c1 should have the same size, and be square"

    # Check for NaN or INF
    assert not (np.any(np.isnan(c0)) or np.any(np.isinf(c0))), "NaN or INF in c0"
    assert not (np.any(np.isnan(c1)) or np.any(np.isinf(c1))), "NaN or INF in c1"

    # derive PCA and whitening matrix from unbiased covariance
    eigvec0, eigval0 = pca(c0, max_comps=keep1, thresh=keep2)

    # apply whitening and PCA matrices to the biased covariance
    # (== covariance of bias whitened data)
    W = np.diag(np.sqrt(1. / eigval0))  # diagonal of whitening matrix

    # c1 is projected into whitened PCA space of data channels
    c2 = (eigvec0 @ W).T @ c1 @ (eigvec0 @ W)

    # proj. matrix from whitened data space to a space maximizing bias
    eigvec2, eigval2 = pca(c2, max_comps=keep1, thresh=keep2)

    # DSS matrix (raw data to normalized DSS)
    todss = eigvec0 @ W @ eigvec2

    # Normalise DSS matrix
    N = np.sqrt(np.diag(todss.T @ c0 @ todss))
    todss /= N

    pwr0 = np.sqrt(np.sum((c0 @ todss) ** 2, axis=0))
    pwr1 = np.sqrt(np.sum((c1 @ todss) ** 2, axis=0))

    # Return data
    # next line equiv. to: np.array([np.dot(todss, ep) for ep in data])
    # dss_data = np.einsum('ij,hjk->hik', todss, data)

    if return_unmixing:
        fromdss = linalg.pinv(todss)
        return todss, fromdss, pwr0, pwr1
    else:
        return todss, pwr0, pwr1


def dss_line(X, fline, sfreq, nremove=1, nfft=1024, nkeep=None, blocksize=None,
             show=False):
    """Apply DSS to remove power line artifacts.

    Implements the ZapLine algorithm described in [1]_.

    Parameters
    ----------
    X : data, shape=(n_samples, n_chans, n_trials)
        Input data.
    fline : float
        Line frequency (normalized to sfreq, if ``sfreq`` == 1).
    sfreq : float
        Sampling frequency (default=1, which assymes ``fline`` is normalised).
    nremove : int
        Number of line noise components to remove (default=1).
    nfft : int
        FFT size (default=1024).
    nkeep : int
        Number of components to keep in DSS (default=None).
    blocksize : int
        If not None (default), covariance is computed on blocks of
        ``blocksize`` samples. This may improve performance for large datasets.
    show: bool
        If True, show DSS results (default=False).

    Returns
    -------
    y : array, shape=(n_samples, n_chans, n_trials)
        Denoised data.
    artifact : array, shape=(n_samples, n_chans, n_trials)
        Artifact

    Examples
    --------
    Apply to X, assuming line frequency=50Hz and sampling rate=1000Hz, plot
    results:
    >>> dss_line(X, 50/1000)

    Removing 4 line-dominated components:
    >>> dss_line(X, 50/1000, 4)

    Truncating PCs beyond the 30th to avoid overfitting:
    >>> dss_line(X, 50/1000, 4, nkeep=30);

    Return cleaned data in y, noise in yy, do not plot:
    >>> [y, artifact] = dss_line(X, 60/1000)

    References
    ----------
    .. [1] de Cheveigné, A. (2019). ZapLine: A simple and effective method to remove
       power line artifacts. NeuroImage, 116356.
       https://doi.org/10.1016/j.neuroimage.2019.116356

    """
    if X.shape[0] < nfft:
        print(f"Reducing nfft to {X.shape[0]}")
        nfft = X.shape[0]
    n_samples, n_chans, _ = theshapeof(X)
    if blocksize is None:
        blocksize = n_samples

    # Recentre data
    X = demean(X, inplace=True)

    # Cancel line_frequency and harmonics + light lowpass
    X_filt = smooth(X, sfreq / fline)

    # X - X_filt results in the artifact plus some residual biological signal
    X_noise = X - X_filt

    # Reduce dimensionality to avoid overfitting
    if nkeep is not None:
        cov_X_res = tscov(X_noise)[0]
        V, _ = pca(cov_X_res, nkeep)
        X_noise_pca = X_noise @ V
    else:
        X_noise_pca = X_noise.copy()
        nkeep = n_chans

    # Compute blockwise covariances of raw and biased data
    n_harm = np.floor((sfreq / 2) / fline).astype(int)
    c0 = np.zeros((nkeep, nkeep))
    c1 = np.zeros((nkeep, nkeep))
    for X_block in sliding_window_view(X_noise_pca, (blocksize, nkeep),
                                       axis=(0, 1))[::blocksize, 0]:
        # if n_trials>1, reshape to (n_samples, nkeep, n_trials)
        if X_block.ndim == 3:
            X_block = X_block.transpose(1, 2, 0)

        # bias data
        c0 += tscov(X_block)[0]
        c1 += tscov(gaussfilt(X_block, sfreq, fline, fwhm=1, n_harm=n_harm))[0]

    # DSS to isolate line components from residual
    todss, _, pwr0, pwr1 = dss0(c0, c1)

    if show:
        import matplotlib.pyplot as plt
        plt.plot(pwr1 / pwr0, ".-")
        plt.xlabel("component")
        plt.ylabel("score")
        plt.title("DSS to enhance line frequencies")
        plt.show()

    # Remove line components from X_noise
    idx_remove = np.arange(nremove)
    X_artifact = matmul3d(X_noise_pca, todss[:, idx_remove])
    X_res = tsr(X_noise, X_artifact)[0]  # project them out

    # reconstruct clean signal
    y = X_filt + X_res

    # Power of components
    p = wpwr(X - y)[0] / wpwr(X)[0]
    print(f"Power of components removed by DSS: {p:.2f}")
    # return the reconstructed clean signal, and the artifact
    return y, X - y


def dss_line_iter(data, fline, sfreq, win_sz=10, spot_sz=2.5,
                  nfft=512, show=False, dirname=None, extension=".png", n_iter_max=100):
    """Remove power line artifact iteratively.

    This method applies dss_line() until the artifact has been smoothed out
    from the spectrum.

    Parameters
    ----------
    data : data, shape=(n_samples, n_chans, n_trials)
        Input data.
    fline : float
        Line frequency.
    sfreq : float
        Sampling frequency.
    win_sz : float
        Half of the width of the window around the target frequency used to fit
        the polynomial (default=10).
    spot_sz : float
        Half of the width of the window around the target frequency used to
        remove the peak and interpolate (default=2.5).
    nfft : int
        FFT size for the internal PSD calculation (default=512).
    show: bool
        Produce a visual output of each iteration (default=False).
    dirname: str
        Path to the directory where visual outputs are saved when show is 'True'.
        If 'None', does not save the outputs. (default=None)
    extension: str
        Extension of the images filenames. Must be compatible with plt.savefig()
        function. (default=".png")
    n_iter_max : int
        Maximum number of iterations (default=100).

    Returns
    -------
    data : array, shape=(n_samples, n_chans, n_trials)
        Denoised data.
    iterations : int
        Number of iterations.
    """

    def nan_basic_interp(array):
        """Nan interpolation."""
        nans, ix = np.isnan(array), lambda x: x.nonzero()[0]
        array[nans] = np.interp(ix(nans), ix(~nans), array[~nans])
        return array

    freq_rn = [fline - win_sz, fline + win_sz]
    freq_sp = [fline - spot_sz, fline + spot_sz]
    freq, psd = welch(data, fs=sfreq, nfft=nfft, axis=0)

    freq_rn_ix = np.logical_and(freq >= freq_rn[0],
                                freq <= freq_rn[1])
    freq_used = freq[freq_rn_ix]
    freq_sp_ix = np.logical_and(freq_used >= freq_sp[0],
                                freq_used <= freq_sp[1])

    if psd.ndim == 3:
        mean_psd = np.mean(psd, axis=(1, 2))[freq_rn_ix]
    elif psd.ndim == 2:
        mean_psd = np.mean(psd, axis=(1))[freq_rn_ix]

    mean_psd_wospot = mean_psd.copy()
    mean_psd_wospot[freq_sp_ix] = np.nan
    mean_psd_tf = nan_basic_interp(mean_psd_wospot)
    pf = np.polyfit(freq_used, mean_psd_tf, 3)
    p = np.poly1d(pf)
    clean_fit_line = p(freq_used)

    aggr_resid = []
    iterations = 0
    while iterations < n_iter_max:
        data, _ = dss_line(data, fline, sfreq, nfft=nfft, nremove=1)
        freq, psd = welch(data, fs=sfreq, nfft=nfft, axis=0)
        if psd.ndim == 3:
            mean_psd = np.mean(psd, axis=(1, 2))[freq_rn_ix]
        elif psd.ndim == 2:
            mean_psd = np.mean(psd, axis=(1))[freq_rn_ix]

        residuals = mean_psd - clean_fit_line
        mean_score = np.mean(residuals[freq_sp_ix])
        aggr_resid.append(mean_score)

        print(f"Iteration {iterations} score: {mean_score}")

        if show:
            import matplotlib.pyplot as plt
            f, ax = plt.subplots(2, 2, figsize=(12, 6), facecolor="white")

            if psd.ndim == 3:
                mean_sens = np.mean(psd, axis=2)
            elif psd.ndim == 2:
                mean_sens = psd

            y = mean_sens[freq_rn_ix]
            ax.flat[0].plot(freq_used, y)
            ax.flat[0].set_title("Mean PSD across trials")
            ax.flat[0].set_xlabel("Frequency (Hz)")
            ax.flat[0].set_ylabel("Power")

            ax.flat[1].plot(freq_used, mean_psd_tf, c="gray",
                            label="Interpolated mean PSD")
            ax.flat[1].plot(freq_used, mean_psd, c="blue", label="Mean PSD")
            ax.flat[1].plot(freq_used, clean_fit_line, c="red", label="Fitted polynomial")
            ax.flat[1].set_title("Mean PSD across trials and sensors")
            ax.flat[1].set_xlabel("Frequency (Hz)")
            ax.flat[1].set_ylabel("Power")
            ax.flat[1].legend()

            tf_ix = np.where(freq_used <= fline)[0][-1]
            ax.flat[2].plot(freq_used, residuals)
            color = "green"
            if mean_score <= 0:
                color = "red"
            ax.flat[2].scatter(freq_used[tf_ix], residuals[tf_ix], c=color)
            ax.flat[2].set_title("Residuals")
            ax.flat[2].set_xlabel("Frequency (Hz)")
            ax.flat[2].set_ylabel("Power")

            ax.flat[3].plot(np.arange(iterations + 1), aggr_resid, marker="o")
            ax.flat[3].set_title("Aggregated residuals")
            ax.flat[3].set_xlabel("Iteration")
            ax.flat[3].set_ylabel("Power")

            plt.tight_layout()
            if dirname is not None:
                plt.savefig(Path(dirname) / f"dss_iter_{iterations:03}{extension}")
            plt.show()

        if mean_score <= 0:
            break

        iterations += 1

    if iterations == n_iter_max:
        raise RuntimeError("Could not converge. Consider increasing the "
                           "maximum number of iterations")

    return data, iterations


def dss_line_plus(
    data: np.ndarray,
    sfreq: float,
    fline: float | list[float] | None = None,
    nkeep: int = 0,
    adaptive_n_remove: bool = True,
    fixed_n_remove: int = 1,
    chunk_length: float = 0.0,
    min_chunk_length: float = 30.0,
    plot_results: bool = False,
    figsize: tuple[int, int] = (14, 10),
    vanilla_mode: bool = False,
    dirname: str | None = None,
    noise_comp_sigma: float = 3.0,
    adaptive_sigma: bool = True,
    min_sigma: float = 2.5,
    max_sigma: float = 4.0,
    min_freq: float = 17.0,
    max_freq: float = 99.0,
    win_sz: float = 6.0,
    coarse_power_diff_hi: float = 4.0,
    coarse_power_diff_lo: float = 1.76,
    detect_chunk_freqs: bool = True,
    fine_threshold_mult: float = 2.0,
    fine_bounds_hi: tuple[float, float] = (0.05, 0.05),
    fine_bounds_lo: tuple[float, float] = (0.4, 0.1),
    max_prop_above_hi: float = 0.005,
    max_prop_below_lo: float = 0.005,
) -> tuple[np.ndarray, dict]:
    """Remove line noise and other frequency-specific artifacts using Zapline-plus.

    Parameters
    ----------
    Core parameters
    ~~~~~~~~~~~~~~~
    data : array, shape=(n_times, n_chans)
        Input data (time, channels).
    sfreq : float
        Sampling frequency in Hz.
    fline : float | list of float | None
        Noise frequency or frequencies to remove. If None, frequencies are
        detected automatically (default=None).
    nkeep : int
        Number of principal components to keep in DSS. If 0, no dimensionality
        reduction is applied (default=0).
    adaptive_n_remove : bool
        Automatically detect the number of components to remove
        (default=True).
    fixed_n_remove : int
        Fixed number of components to remove per chunk. Used when
        adaptive_n_remove=False, or as minimum when adaptive_n_remove=True
        (default=1).
    chunk_length : float
        Length of chunks in seconds. If 0, adaptive chunking is used.
        If -1, process the entire recording as a single chunk
        (default=0.0).
    min_chunk_length : float
        Minimum chunk length in seconds when using adaptive chunking
        (default=30.0).
    plot_results : bool
        If True, generate diagnostic plots for each cleaned frequency
        (default=False).
    figsize : tuple of int
        Figure size for diagnostic plots (default=(14, 10)).
    vanilla_mode : bool
        If True, disable all Zapline-plus features and use vanilla Zapline
        behavior (default=False).
    dirname : str | None
        Path to directory where visual outputs are saved when plot_results is True
        (default=None).

    Sigma / component detection parameters
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    noise_comp_sigma : float
        Initial SD threshold for iterative outlier detection of noise
        components (default=3.0).
    adaptive_sigma : bool
        Automatically adapt noise_comp_sigma and fixed_n_remove based on
        cleaning results (default=True).
    min_sigma : float
        Minimum SD threshold when adapting noise_comp_sigma (default=2.5).
    max_sigma : float
        Maximum SD threshold when adapting noise_comp_sigma (default=4.0).

    Frequency detection parameters
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    min_freq : float
        Minimum frequency (Hz) considered for automatic noise detection
        (default=17.0).
    max_freq : float
        Maximum frequency (Hz) considered for automatic noise detection
        (default=99.0).
    win_sz : float
        Window size in Hz for noise frequency detection (default=6.0).
    coarse_power_diff_hi : float
        Power threshold (10*log10) above center power to detect the start
        of a noise peak (default=4.0).
    coarse_power_diff_lo : float
        Power threshold (10*log10) above center power to detect the end
        of a noise peak (default=1.76).
    detect_chunk_freqs : bool
        If True, detect individual noise frequencies within each chunk
        (default=True).
    fine_threshold_mult : float
        Multiplier for fine-grained noise frequency detection threshold
        (default=2.0).

    Cleaning quality evaluation
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fine_bounds_hi : tuple of float
        Frequency bounds (Hz) around the target frequency used to detect
        insufficient cleaning (default=(0.05, 0.05)).
    fine_bounds_lo : tuple of float
        Frequency bounds (Hz) around the target frequency used to detect
        excessive cleaning (default=(0.4, 0.1)).
    max_prop_above_hi : float
        Maximum proportion of samples above the upper fine threshold before
        cleaning is considered too weak (default=0.005).
    max_prop_below_lo : float
        Maximum proportion of samples below the lower fine threshold before
        cleaning is considered too strong (default=0.005).


    Returns
    -------
    clean_data : array, shape=(n_times, n_chans)
        Cleaned data.
    config : dict
        Configuration dictionary containing all parameters and analytics.

    Notes
    -----
    The algorithm proceeds as follows:
    1. Detect noise frequencies (if not provided)
    2. Segment data into chunks with stable noise topography
    3. Apply Zapline to each chunk
    4. Automatically detect and remove noise components
    5. Adapt parameters if cleaning is too weak or too strong

    Examples
    --------
    Remove 50 Hz line noise automatically:
    >>> clean_data, config = dss_line_plus(data, sfreq=500, fline=50)

    Remove line noise with automatic frequency detection:
    >>> clean_data, config = dss_line_plus(data, sfreq=500)

    References
    ----------
    .. [1] Klug, M., & Kloosterman, N. A. (2022). Zapline-plus: A Zapline extension for
       automatic and adaptive removal of frequency-specific noise artifacts in M/EEG.
       Human Brain Mapping, 43(9), 2743-2758.
       https://doi.org/10.1002/hbm.25832

    .. [2] Klug, M. (2023). Zapline-Plus GitHub repository.
       https://github.com/MariusKlug/zapline-plus

    """
    n_times, n_chans = data.shape

    # handle vanilla mode (ZapLine without plus)
    if vanilla_mode:
        logging.warning(
            "vanilla_mode=True: Using vanilla Zapline behavior. "
            "All adaptive features disabled."
        )
        if fline is None:
            raise ValueError("vanilla_mode requires fline to be specified (not None)")

        for param_name in [
            "adaptive_n_remove",
            "adaptive_sigma",
            "detect_chunk_freqs",
        ]:
            if locals()[param_name]:
                logging.warning(f"vanilla_mode=True: Overriding {param_name} to False.")

        # Override all adaptive features
        adaptive_n_remove = False
        adaptive_sigma = False
        detect_chunk_freqs = False
        chunk_length = -1  # Zapline vanilla deals with single chunk

    # if nothing is adaptive, only one iteration per frequency
    if not (adaptive_n_remove and adaptive_sigma):
        max_iterations = 1

    # check for globally flat channels
    # will be omitted during processing and reintroduced later
    diff_data = np.diff(data, axis=0)
    global_flat = np.where(np.all(diff_data == 0, axis=0))[0]
    if len(global_flat) > 0:
        logging.warning(
            f"Detected {len(global_flat)} globally flat channels: {global_flat}. "
            f"Removing for processing, will add back after."
        )
        flat_data = data[:, global_flat]
        active_channels = np.setdiff1d(np.arange(n_chans), global_flat)
        data = data[:, active_channels]
    else:
        active_channels = np.arange(n_chans)
        flat_data = None

    # initialize configuration
    config = {
        key: locals()[key]
        for key in (
            "sfreq",
            "fline",
            "nkeep",
            "adaptive_n_remove",
            "fixed_n_remove",
            "min_freq",
            "max_freq",
            "chunk_length",
            "min_chunk_length",
            "noise_comp_sigma",
            "adaptive_sigma",
            "min_sigma",
            "max_sigma",
            "win_sz",
            "coarse_power_diff_hi",
            "coarse_power_diff_lo",
            "detect_chunk_freqs",
            "fine_threshold_mult",
            "fine_bounds_hi",
            "fine_bounds_lo",
            "max_prop_above_hi",
            "max_prop_below_lo",
        )
    }
    config["analytics"] = {}


    # detect noise frequencies if not provided
    if fline is None:
        fline = _detect_noise_frequencies(
            data,
            sfreq,
            min_freq,
            max_freq,
            win_sz,
            coarse_power_diff_hi,
            coarse_power_diff_lo,
        )
    elif not isinstance(fline, list):
        fline = [fline]

    if len(fline) == 0:
        logging.info("No noise frequencies detected. Returning original data.")
        return data.copy(), config

    config["detected_fline"] = fline

    # retain input data
    cleaned_data = data.copy()

    # Process each noise frequency
    for freq_idx, target_freq in enumerate(fline):
        print(f"Processing noise frequency: {target_freq:.2f} Hz")

        if chunk_length == -1:
            # single chunk
            chunk_bounds = [(0, n_times)]
        elif chunk_length == 0:
            # adaptive chunking
            chunk_bounds = _adaptive_chunking(
                cleaned_data,
                sfreq,
                target_freq,
                min_chunk_length
                )
        else:
            # fixed-length chunks
            chunk_samples = int(chunk_length * sfreq)
            chunk_bounds = [
                (i, min(i + chunk_samples, n_times))
                for i in range(0, n_times, chunk_samples)
            ]

        # initialize tracking variables
        current_sigma = noise_comp_sigma
        current_n_remove = fixed_n_remove
        seen_too_strong = False
        iteration_idx = 0
        max_iterations = 20

        while iteration_idx < max_iterations:
            iteration_idx += 1

            # Clean each chunk
            chunk_results = []
            for chunk_start, chunk_end in chunk_bounds:
                chunk_data = cleaned_data[chunk_start:chunk_end, :]

                # Detect chunk-specific noise frequency
                if detect_chunk_freqs:
                    chunk_freq, has_noise = _detect_chunk_noise_frequency(
                        chunk_data,
                        sfreq,
                        target_freq,
                        win_sz,
                        fine_threshold_mult,
                        detailed_freq_bounds=fine_bounds_hi,
                    )
                else:
                    chunk_freq = target_freq
                    has_noise = True

                # Apply Zapline to chunk
                if has_noise:
                    if adaptive_n_remove:
                        n_remove = _detect_noise_components(
                            chunk_data, sfreq, chunk_freq, current_sigma, nkeep
                        )
                        n_remove = max(n_remove, current_n_remove)
                    else:
                        n_remove = current_n_remove

                    # Cap at 1/5 of components
                    n_remove = min(n_remove, n_chans // 5)
                else:
                    n_remove = current_n_remove

                # clean chunk
                cleaned_chunk = _apply_zapline_to_chunk(
                    chunk_data, sfreq, chunk_freq, n_remove, nkeep
                )

                chunk_results.append(
                    {
                        "start": chunk_start,
                        "end": chunk_end,
                        "freq": chunk_freq,
                        "n_remove": n_remove,
                        "has_noise": has_noise,
                        "data": cleaned_chunk,
                    }
                )

            # reconstruct cleaned data
            cleaned_candidate = cleaned_data.copy()
            for result in chunk_results:
                cleaned_candidate[result["start"] : result["end"], :] = result["data"]

            # check if cleaning is optimal
            cleaning_status = _check_cleaning_quality(
                data,
                cleaned_candidate,
                sfreq,
                target_freq,
                win_sz,
                fine_threshold_mult,
                fine_bounds_hi,
                fine_bounds_lo,
                max_prop_above_hi,
                max_prop_below_lo,
            )

            # store analytics
            config["analytics"][f"freq_{freq_idx}"] = {
                "target_freq": target_freq,
                "iteration": iteration_idx,
                "sigma": current_sigma,
                "fixed_nremove": current_n_remove,
                "n_chunks": len(chunk_bounds),
                "chunk_results": chunk_results,
                "cleaning_status": cleaning_status,
            }

            # check if we need to adapt
            if cleaning_status == "good":
                cleaned_data = cleaned_candidate
                break

            elif cleaning_status == "too_weak" and not seen_too_strong:
                if current_sigma > min_sigma:
                    current_sigma = max(current_sigma - 0.25, min_sigma)
                    current_n_remove += 1
                    logging.info(
                        f"Cleaning too weak. Adjusting sigma to {current_sigma:.2f}, "
                        f"fixed removal to {current_n_remove}"
                    )
                else:
                    logging.info("At minimum sigma, accepting result")
                    cleaned_data = cleaned_candidate
                    break

            elif cleaning_status == "too_strong":
                seen_too_strong = True
                if current_sigma < max_sigma:
                    current_sigma = min(current_sigma + 0.25, max_sigma)
                    current_n_remove = max(current_n_remove - 1, fixed_n_remove)
                    logging.info(
                        f"Cleaning too strong. Adjusting sigma to {current_sigma:.2f}, "
                        f"fixed removal to {current_n_remove}"
                    )
                else:
                    logging.info("At maximum sigma, accepting result")
                    cleaned_data = cleaned_candidate
                    break

            else:
                # Too strong takes precedence, or we can't improve further
                cleaned_data = cleaned_candidate
                break

        # Generate diagnostic plot
        if plot_results:
            _plot_cleaning_results(
                data,
                cleaned_data,
                sfreq,
                target_freq,
                config["analytics"][f"freq_{freq_idx}"],
                figsize,
                dirname,
            )

    # add flat channels back to data, if present
    if flat_data is not None:
        full_clean = np.zeros((n_times, n_chans))
        full_clean[:, active_channels] = cleaned_data
        full_clean[:, global_flat] = flat_data
        cleaned_data = full_clean

    return cleaned_data, config


def _detect_noise_frequencies(
    data, sfreq, minfreq, maxfreq, winsize, power_diff_high, power_diff_low
):
    """
    Detect noise frequencies.

    This is an exact implementation of find_next_noisefreq.m with the only difference
    that all peaks are returned instead of this being called iteratively.

    How it works
    ------------
    1. Compute PSD and log-transform.
    2. Slide a window across frequencies from minfreq to maxfreq.
    3. For each frequency, compute center power as mean of left and right thirds.
    4. Use a state machine to detect peaks:
        - SEARCHING: If current power - center power > power_diff_high,
            mark peak start and switch to IN_PEAK.
        - IN_PEAK: If current power - center power <= power_diff_low,
            mark peak end, find max within peak, record frequency,
            and switch to SEARCHING.
    5. Return list of detected noise frequencies.
    """
    # Compute PSD
    freqs, psd = _compute_psd(data, sfreq)
    log_psd = 10 * np.log10(np.mean(psd, axis=1))

    # State machine variables
    in_peak = False
    peak_start_idx = None
    noise_freqs = []

    # Search bounds
    start_idx = np.searchsorted(freqs, minfreq)
    end_idx = np.searchsorted(freqs, maxfreq)

    # Window size in samples
    freq_resolution = freqs[1] - freqs[0]
    win_samples = int(winsize / freq_resolution)

    idx = start_idx
    while idx < end_idx:
        # Get window around current frequency
        win_start = max(0, idx - win_samples // 2)
        win_end = min(len(freqs), idx + win_samples // 2)
        win_psd = log_psd[win_start:win_end]

        if len(win_psd) < 3:
            idx += 1
            continue

        # Compute center power (mean of left and right thirds)
        n_third = len(win_psd) // 3
        if n_third < 1:
            idx += 1
            continue

        left_third = win_psd[:n_third]
        right_third = win_psd[-n_third:]
        center_power = np.mean([np.mean(left_third), np.mean(right_third)])

        current_power = log_psd[idx]

        # State machine logic
        if not in_peak:
            # State: SEARCHING - Check for peak start
            if current_power - center_power > power_diff_high:
                in_peak = True
                peak_start_idx = idx

        else:
            # State: IN_PEAK - Check for peak end
            if current_power - center_power <= power_diff_low:
                in_peak = False
                peak_end_idx = idx

                # Find the actual maximum within the peak
                if peak_start_idx is not None and peak_end_idx > peak_start_idx:
                    peak_region = log_psd[peak_start_idx:peak_end_idx]
                    max_offset = np.argmax(peak_region)
                    max_idx = peak_start_idx + max_offset
                    noise_freqs.append(freqs[max_idx])

                    # Skip past this peak to avoid re-detection
                    idx = peak_end_idx
                    continue

        idx += 1

    return noise_freqs


def _adaptive_chunking(
    data,
    sfreq,
    target_freq,
    min_chunk_length,
    detection_winsize=6.0,
    prominence_quantile=0.95,
):
    """Segment data into chunks with stable noise topography."""
    n_times, n_chans = data.shape

    if n_times < sfreq * min_chunk_length:
        logging.warning("Data too short for adaptive chunking. Using single chunk.")
        return [(0, n_times)]

    # Narrow-band filter around target frequency
    bandwidth = detection_winsize / 2.0
    filtered = _narrowband_filter(data, sfreq, target_freq, bandwidth=bandwidth)

    # Compute covariance matrices for 1-second epochs
    epoch_length = int(sfreq)
    n_epochs = n_times // epoch_length

    distances = np.zeros(n_epochs)
    prev_cov = None

    for i in range(n_epochs):
        start = i * epoch_length
        end = start + epoch_length
        epoch = filtered[start:end, :]
        cov = np.cov(epoch, rowvar=False)

        if prev_cov is not None:
            # Frobenius norm of difference
            distances[i] = np.linalg.norm(cov - prev_cov, "fro")
        # else: distance[i] already 0 from initialization

        prev_cov = cov

    if len(distances) < 2:
        return [(0, n_times)]

    # find all peaks to get prominence distribution
    peaks_all, properties_all = find_peaks(distances, prominence=0)

    if len(peaks_all) == 0 or "prominences" not in properties_all:
        # No peaks found
        logging.warning("No peaks found in distance signal. Using single chunk.")
        return [(0, n_times)]

    prominences = properties_all["prominences"]

    # filter by prominence quantile
    min_prominence = np.quantile(prominences, prominence_quantile)
    min_distance_epochs = int(min_chunk_length)  # Convert seconds to epochs

    peaks, properties = find_peaks(
        distances, prominence=min_prominence, distance=min_distance_epochs
    )

    # convert peak locations (in epochs) to sample indices
    chunk_starts = [0]
    for peak in peaks:
        chunk_start_sample = peak * epoch_length
        chunk_starts.append(chunk_start_sample)
    chunk_starts.append(n_times)

    # create chunk list
    chunks = []
    for i in range(len(chunk_starts) - 1):
        start = chunk_starts[i]
        end = chunk_starts[i + 1]
        chunks.append((start, end))

    # ensure minimum chunk length at edges
    min_chunk_samples = int(min_chunk_length * sfreq)

    if len(chunks) > 1:
        # check first chunk
        if chunks[0][1] - chunks[0][0] < min_chunk_samples:
            # merge with next
            chunks[1] = (chunks[0][0], chunks[1][1])
            chunks.pop(0)

    if len(chunks) > 1:
        # check last chunk
        if chunks[-1][1] - chunks[-1][0] < min_chunk_samples:
            # merge with previous
            chunks[-2] = (chunks[-2][0], chunks[-1][1])
            chunks.pop(-1)

    return chunks


def _detect_chunk_noise_frequency(
    data,
    sfreq,
    target_freq,
    winsize,
    mult_fine,
    detailed_freq_bounds=(-0.05, 0.05),  # ← Add this parameter
):
    """Detect chunk-specific noise frequency around target."""
    freqs, psd = _compute_psd(data, sfreq)
    log_psd = 10 * np.log10(np.mean(psd, axis=1))

    # get frequency mask
    search_mask = (freqs >= target_freq + detailed_freq_bounds[0]) & (
        freqs <= target_freq + detailed_freq_bounds[1]
    )

    if not np.any(search_mask):
        return target_freq, False

    search_freqs = freqs[search_mask]
    search_psd = log_psd[search_mask]

    # find peak
    peak_idx = np.argmax(search_psd)
    peak_freq = search_freqs[peak_idx]
    peak_power = search_psd[peak_idx]

    # Compute threshold (uses broader window)
    win_mask = (freqs >= target_freq - winsize / 2) & (freqs <= target_freq + winsize / 2)
    win_psd = log_psd[win_mask]

    n_third = len(win_psd) // 3
    left_third = win_psd[:n_third]
    right_third = win_psd[-n_third:]
    center = np.mean([np.mean(left_third), np.mean(right_third)])

    # Compute deviation (lower 5% quantiles)
    lower_quant_left = np.percentile(left_third, 5)
    lower_quant_right = np.percentile(right_third, 5)
    deviation = center - np.mean([lower_quant_left, lower_quant_right])

    threshold = center + mult_fine * deviation

    has_noise = peak_power > threshold

    return peak_freq, has_noise


def _detect_noise_components(data, sfreq, target_freq, sigma, nkeep):
    """Detect number of noise components to remove using outlier detection."""
    # Convert nkeep=0 to None for dss_line (0 means no reduction)
    if nkeep == 0:
        nkeep = None

    # Apply DSS to get component scores
    _, scores = dss_line(data, target_freq, sfreq, nkeep=nkeep)

    if scores is None or len(scores) == 0:
        return 1

    # Sort scores in descending order
    sorted_scores = np.sort(scores)[::-1]

    # Iterative outlier detection
    n_remove = 0
    remaining = sorted_scores.copy()

    while len(remaining) > 1:
        mean_val = np.mean(remaining)
        std_val = np.std(remaining)
        threshold = mean_val + sigma * std_val

        if remaining[0] > threshold:
            n_remove += 1
            remaining = remaining[1:]
        else:
            break

    return max(n_remove, 1)


def _apply_zapline_to_chunk(chunk_data, sfreq, chunk_freq, n_remove, nkeep):
    """Apply Zapline to a single chunk, handling flat channels."""
    n_samples, n_chans = chunk_data.shape

    # Convert nkeep=0 to None for dss_line (0 means no reduction)
    if nkeep == 0:
        nkeep = None

    # Detect flat channels (zero variance)
    diff_chunk = np.diff(chunk_data, axis=0)
    flat_channels = np.where(np.all(diff_chunk == 0, axis=0))[0]

    if len(flat_channels) > 0:
        logging.warning(
            f"Detected {len(flat_channels)} flat channels in chunk: {flat_channels}. "
            f"Removing temporarily for processing."
        )

        # store flat channel data
        flat_channel_data = chunk_data[:, flat_channels]

        # remove flat channels from processing
        active_channels = np.setdiff1d(np.arange(n_chans), flat_channels)
        chunk_data_active = chunk_data[:, active_channels]

        # process only active channels
        cleaned_active, _ = dss_line(
            chunk_data_active,
            fline=chunk_freq,
            sfreq=sfreq,
            nremove=n_remove,
            nkeep=nkeep,
        )

        # Reconstruct full data with flat channels
        cleaned_chunk = np.zeros_like(chunk_data)
        cleaned_chunk[:, active_channels] = cleaned_active
        cleaned_chunk[:, flat_channels] = (
            flat_channel_data  # Add flat channels back unchanged
        )

    else:
        # no flat channels, process normally
        cleaned_chunk, _ = dss_line(
            chunk_data,
            fline=chunk_freq,
            sfreq=sfreq,
            nremove=n_remove,
            nkeep=nkeep,
        )

    return cleaned_chunk


def _check_cleaning_quality(
    original_data,
    cleaned_data,
    sfreq,
    target_freq,
    winsize,
    mult_fine,
    bounds_upper,
    bounds_lower,
    max_prop_above,
    max_prop_below,
):
    """Check if cleaning is too weak, too strong, or good."""
    # Compute PSDs
    freqs, psd_clean = _compute_psd(cleaned_data, sfreq)
    log_psd_clean = 10 * np.log10(np.mean(psd_clean, axis=1))

    # Compute fine thresholds
    win_mask = (freqs >= target_freq - winsize / 2) & (freqs <= target_freq + winsize / 2)
    win_psd = log_psd_clean[win_mask]

    n_third = len(win_psd) // 3
    left_third = win_psd[:n_third]
    right_third = win_psd[-n_third:]
    center = np.mean([np.mean(left_third), np.mean(right_third)])

    # Deviation from lower quantiles
    lower_quant_left = np.percentile(left_third, 5)
    lower_quant_right = np.percentile(right_third, 5)
    deviation = center - np.mean([lower_quant_left, lower_quant_right])

    # Upper threshold (too weak cleaning)
    upper_mask = (freqs >= target_freq - bounds_upper[0]) & (
        freqs <= target_freq + bounds_upper[1]
    )
    upper_threshold = center + mult_fine * deviation
    upper_psd = log_psd_clean[upper_mask]
    prop_above = np.mean(upper_psd > upper_threshold)

    # Lower threshold (too strong cleaning)
    lower_mask = (freqs >= target_freq - bounds_lower[0]) & (
        freqs <= target_freq + bounds_lower[1]
    )
    lower_threshold = center - mult_fine * deviation
    lower_psd = log_psd_clean[lower_mask]
    prop_below = np.mean(lower_psd < lower_threshold)

    if prop_below > max_prop_below:
        return "too_strong"
    elif prop_above > max_prop_above:
        return "too_weak"
    else:
        return "good"


def _compute_psd(data, sfreq, nperseg=None):
    """Compute power spectral density using Welch's method."""
    if nperseg is None:
        nperseg = int(sfreq * 4)  # 4-second windows

    freqs, psd = welch(
        data,
        fs=sfreq,
        window="hann",
        nperseg=nperseg,
        axis=0,
    )

    return freqs, psd


def _narrowband_filter(data, sfreq, center_freq, bandwidth=3.0):
    """Apply narrow-band filter around center frequency."""
    nyq = sfreq / 2
    low = (center_freq - bandwidth) / nyq
    high = (center_freq + bandwidth) / nyq

    # Ensure valid frequency range
    low = max(low, 0.001)
    high = min(high, 0.999)

    sos = butter(4, [low, high], btype="band", output="sos")
    filtered = sosfiltfilt(sos, data, axis=0)

    return filtered


def _plot_cleaning_results(
    original,
    cleaned,
    sfreq,
    target_freq,
    analytics,
    figsize=(16, 10),
    dirname=None,
):
    """
    Generate diagnostic plots for cleaning results.

    Matches the MATLAB figure style with 8 subplots.

    Parameters
    ----------
    original : ndarray, shape (n_times, n_chans)
        Original data
    cleaned : ndarray, shape (n_times, n_chans)
        Cleaned data
    sfreq : float
        Sampling frequency
    target_freq : float
        Target noise frequency
    analytics : dict
        Analytics from cleaning
    figsize : tuple
        Figure size
    dirname : str, optional
        Directory to save figure
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.35)

    # Compute PSDs
    freqs, psd_orig = _compute_psd(original, sfreq)
    _, psd_clean = _compute_psd(cleaned, sfreq)

    # Compute removed power
    _, psd_removed = _compute_psd(original - cleaned, sfreq)

    # Average across channels and convert to dB
    log_psd_orig = 10 * np.log10(np.mean(psd_orig, axis=1))
    log_psd_clean = 10 * np.log10(np.mean(psd_clean, axis=1))
    log_psd_removed = 10 * np.log10(np.mean(psd_removed, axis=1))

    # ============================================================================
    # A. PREDEFINED FREQUENCY - Zoomed spectrum around noise
    # ============================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    zoom_mask = (freqs >= target_freq - 1) & (freqs <= target_freq + 1)
    ax1.plot(freqs[zoom_mask], log_psd_orig[zoom_mask], "k-", linewidth=1.5)
    ax1.set_xlabel("Frequency [Hz]")
    ax1.set_ylabel(r"Power [$10 \cdot \log_{10}$ V²/Hz]")
    ax1.set_title(f"A   predefined frequency:\n      {target_freq:.0f}Hz")
    ax1.grid(True, alpha=0.3)

    # ============================================================================
    # B. REMOVED COMPONENTS - Bar plot per chunk
    # ============================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    chunk_results = analytics["chunk_results"]
    n_removes = [cr["n_remove"] for cr in chunk_results]
    n_chunks = len(n_removes)
    mean_removed = np.mean(n_removes)

    ax2.bar(range(n_chunks), n_removes, color="gray", edgecolor="black", linewidth=0.5)
    ax2.set_xlabel("Chunk")
    ax2.set_ylabel("# Components removed")
    ax2.set_title(f"B   # removed comps in {n_chunks} chunks, μ = {mean_removed:.2f}")
    ax2.grid(True, alpha=0.3, axis="y")

    # ============================================================================
    # C. INDIVIDUAL NOISE FREQUENCIES - Time series
    # ============================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    chunk_freqs = [cr["freq"] for cr in chunk_results]
    time_min = np.array([cr["start"] for cr in chunk_results]) / sfreq / 60

    ax3.plot(time_min, chunk_freqs, "o-", color="gray", linewidth=1, markersize=4)
    ax3.set_xlabel("Time [minutes]")
    ax3.set_ylabel("Frequency [Hz]")
    ax3.set_title("C   individual noise frequencies [Hz]")
    ax3.grid(True, alpha=0.3)

    # Set y-limits with some padding
    freq_range = max(chunk_freqs) - min(chunk_freqs)
    if freq_range > 0:
        padding = freq_range * 0.2
        ax3.set_ylim([min(chunk_freqs) - padding, max(chunk_freqs) + padding])

    # ============================================================================
    # D. MEAN ARTIFACT SCORES - Would need DSS component scores
    # ============================================================================
    ax4 = fig.add_subplot(gs[0, 3])

    # Extract sigma from analytics
    sigma = analytics.get("sigma", 3.0)

    # Placeholder - would need actual component scores from DSS
    # In MATLAB, these come from the DSS decomposition
    ax4.text(
        0.5, 0.5,
        f"D   mean artifact scores [a.u.]\n      sigma for detection = {sigma:.2f}\n\n"
        "(requires DSS component output)",
        ha="center",
        va="center",
        transform=ax4.transAxes,
        fontsize=10,
    )
    ax4.set_title("D   mean artifact scores [a.u.]")
    ax4.set_xlim([0, 1])
    ax4.set_ylim([0, 1])
    ax4.axis("off")

    # ============================================================================
    # E. CLEANED SPECTRUM - Zoomed
    # ============================================================================
    ax5 = fig.add_subplot(gs[1, 0])

    # Compute percentage change
    pct_above = 0.0
    pct_below = 0.0

    ax5.plot(freqs[zoom_mask], log_psd_clean[zoom_mask], color="teal", linewidth=1.5)
    ax5.set_xlabel("Frequency [Hz]")
    ax5.set_ylabel(r"Power [$10 \cdot \log_{10}$ V²/Hz]")
    ax5.set_title(f"E   cleaned spectrum\n      {pct_above:.0f}% above\n"
                  f"      {pct_below:.0f}% below")
    ax5.grid(True, alpha=0.3)

    # ============================================================================
    # F. FULL SPECTRUM - Original with noise peaks
    # ============================================================================
    ax6 = fig.add_subplot(gs[1, 1])

    # Plot full spectrum
    ax6.plot(freqs, log_psd_orig, "k-", linewidth=1, alpha=0.7, label="raw data")

    # Mark noise frequency and harmonics
    ax6.axvline(target_freq, color="red", linestyle="-", linewidth=2, alpha=0.7)

    # Compute noise ratio (noise power / surrounding power)
    noise_mask = (freqs >= target_freq - 0.05) & (freqs <= target_freq + 0.05)

    # Surrounding region (excluding ±1 Hz around noise)
    surround_mask = ((freqs >= target_freq - 5) & (freqs < target_freq - 1)) | \
                    ((freqs > target_freq + 1) & (freqs <= target_freq + 5))

    if np.any(noise_mask) and np.any(surround_mask):
        noise_power = np.mean(psd_orig[noise_mask, :])
        surround_power = np.mean(psd_orig[surround_mask, :])
        noise_ratio = noise_power / surround_power if surround_power > 0 else 0
    else:
        noise_ratio = 0

    ax6.set_xlabel("Frequency [Hz]")
    ax6.set_ylabel(r"Power [$10 \cdot \log_{10}$ V²/Hz]")
    ax6.set_title(f"F   noise frequency: {target_freq:.0f}Hz\n"
                  f"      ratio of noise to surroundings: {noise_ratio:.4f}")
    ax6.set_xlim([0, min(sfreq/2, 200)])
    ax6.grid(True, alpha=0.3)
    ax6.legend(loc="upper right")

    # ============================================================================
    # G. REMOVED POWER - Cleaned vs Removed
    # ============================================================================
    ax7 = fig.add_subplot(gs[1, 2])

    # Compute proportions removed
    total_power_orig = np.mean(psd_orig)
    total_power_removed = np.mean(psd_removed)

    noise_power_orig = np.mean(psd_orig[noise_mask, :]) if np.any(noise_mask) else 0
    noise_power_removed = np.mean(psd_removed[noise_mask, :]) if np.any(noise_mask) else 0

    # Proportion removed at noise frequency
    if noise_power_orig > 0:
        prop_removed_noise = noise_power_removed / noise_power_orig * 100
    else:
        prop_removed_noise = 0

    # Ratio after cleaning
    if np.any(noise_mask):
        noise_power_clean = np.mean(psd_clean[noise_mask, :])
    else:
        noise_power_clean = 0

    if np.any(surround_mask):
        surround_power_clean = np.mean(psd_clean[surround_mask, :])
    else:
        surround_power_clean = 0

    if surround_power_clean > 0:
        noise_ratio_clean = noise_power_clean / surround_power_clean
    else:
        noise_ratio_clean = 0

    # Plot relative to noise frequency
    # Frequency relative to noise
    freq_relative = freqs - target_freq
    rel_mask = (freq_relative >= -3.5) & (freq_relative <= 3.5)

    ax7.plot(freq_relative[rel_mask], log_psd_clean[rel_mask],
             color="teal", linewidth=1.5, label="clean data")
    ax7.plot(freq_relative[rel_mask], log_psd_removed[rel_mask],
             color="orange", linewidth=1.5, label="removed data", alpha=0.7)

    ax7.set_xlabel("Frequency (relative to noise)")
    ax7.set_ylabel(r"Power [$10 \cdot \log_{10}$ V²/Hz]")
    ax7.set_title(f"G   removed power at noise frequency: {prop_removed_noise:.4f}%\n"
                  f"      ratio of noise to surroundings: {noise_ratio_clean:.4f}")
    ax7.legend(loc="upper right")
    ax7.grid(True, alpha=0.3)

    # ============================================================================
    # H. POWER BELOW NOISE - Comparison
    # ============================================================================
    ax8 = fig.add_subplot(gs[1, 3])

    # Below noise frequency (e.g., target_freq - 11 to target_freq - 1)
    below_mask = (freqs >= max(0, target_freq - 11)) & (freqs <= target_freq - 1)
    # Compute proportions
    if np.any(below_mask):
        below_power_orig = np.mean(psd_orig[below_mask, :])
        below_power_removed = np.mean(psd_removed[below_mask, :])
    else:
        below_power_orig = 0
        below_power_removed = 0

    if total_power_orig > 0:
        prop_removed_full = (total_power_removed / total_power_orig * 100)
    else:
        prop_removed_full = 0
    if below_power_orig > 0:
        prop_removed_below = (below_power_removed / below_power_orig * 100)
    else:
        prop_removed_below = 0

    ax8.plot(freqs[below_mask], log_psd_orig[below_mask],
             "k-", linewidth=1, alpha=0.7, label="raw data")
    ax8.plot(freqs[below_mask], log_psd_clean[below_mask],
             color="teal", linewidth=1.5, label="clean data")

    ax8.set_xlabel("Frequency [Hz]")
    ax8.set_ylabel(r"Power [$10 \cdot \log_{10}$ V²/Hz]")
    ax8.set_title(f"H   removed of full spectrum: {prop_removed_full:.4f}%\n"
                  f"      removed below noise: {prop_removed_below:.4f}%")
    ax8.legend(loc="upper right")
    ax8.grid(True, alpha=0.3)

    plt.suptitle(
        f"Zapline-plus cleaning results: {target_freq:.2f} Hz "
        f"(iteration {analytics['iteration']})",
        fontsize=14,
        y=0.995,
    )
    plt.tight_layout()

    if dirname is not None:
        import os
        os.makedirs(dirname, exist_ok=True)
        filepath = os.path.join(dirname, f"zapline_plus_{target_freq:.0f}Hz.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"Saved figure to: {filepath}")

    return fig
