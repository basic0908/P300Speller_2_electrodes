import numpy as np
import joblib
from scipy.signal import butter, filtfilt
import os
import sys

# =============================================================================
# Helper Functions (verified against Load_Viedataset.ipynb)
# =============================================================================

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    """Applies a Butterworth band-pass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    meandat = np.mean(data, axis=1, keepdims=True)
    data = data - meandat
    y = filtfilt(b, a, data, axis=1)
    return y

def extractEpoch3D(data, event_samples, srate, baseline_ms, frame_ms, opt_keep_baseline):
    """
    Extracts and baseline-corrects time-locked epochs.
    """
    if opt_keep_baseline:
        begin_epoch_sample_offset = int(np.floor(baseline_ms[0] / 1000 * srate))
        end_epoch_sample_offset = int(np.floor(frame_ms[1] / 1000 * srate))
    else:
        begin_epoch_sample_offset = int(np.floor(frame_ms[0] / 1000 * srate))
        end_epoch_sample_offset = int(np.floor(frame_ms[1] / 1000 * srate))
    
    epoch_length_samples = end_epoch_sample_offset - begin_epoch_sample_offset
    if epoch_length_samples <= 0:
        raise ValueError("Calculated epoch length is zero or negative.")

    epoch3D = np.zeros((data.shape[0], epoch_length_samples, len(event_samples)))
    nth_event = 0

    for i_event_sample in event_samples:
        epoch_start_idx = int(i_event_sample + begin_epoch_sample_offset)
        epoch_end_idx = int(i_event_sample + end_epoch_sample_offset)
        if epoch_start_idx < 0 or epoch_end_idx > data.shape[1]:
            continue

        tmp_epoch_data = data[:, epoch_start_idx:epoch_end_idx]
        base_start_idx_orig = int(i_event_sample + np.floor(baseline_ms[0] / 1000 * srate))
        base_end_idx_orig = int(i_event_sample + np.floor(baseline_ms[1] / 1000 * srate))

        if base_start_idx_orig < 0 or base_end_idx_orig >= data.shape[1] or (base_end_idx_orig - base_start_idx_orig) <= 0:
            continue

        baseline_mean = np.mean(data[:, base_start_idx_orig:base_end_idx_orig], axis=1, keepdims=True)
        rmbase_data = tmp_epoch_data - baseline_mean

        if rmbase_data.shape[1] == epoch_length_samples:
            epoch3D[:, :, nth_event] = rmbase_data
            nth_event += 1

    return epoch3D[:, :, :nth_event]


def decimation_by_avg(data, factor):
    """Downsamples 3D data [ch, time, trial] by averaging."""
    n_ch, n_frame, n_trial = data.shape
    decimated_frame = int(np.floor(n_frame / factor))
    decimated_data = np.zeros((n_ch, decimated_frame, n_trial))

    for i in range(n_trial):
        for j in range(decimated_frame):
            start = j * factor
            end = (j + 1) * factor
            decimated_data[:, j, i] = np.mean(data[:, start:end, i], axis=1)

    return decimated_data

def create_speller_grid():
    """Creates the 6x6 P300 speller grid."""
    return np.array([
        ['A', 'B', 'C', 'D', 'E', 'F'],
        ['G', 'H', 'I', 'J', 'K', 'L'],
        ['M', 'N', 'O', 'P', 'Q', 'R'],
        ['S', 'T', 'U', 'V', 'W', 'X'],
        ['Y', 'Z', '0', '1', '2', '3'],
        ['4', '5', '6', '7', '8', '9']
    ])

# =============================================================================
# Main Prediction Function
# =============================================================================

def predict_speller_char(eeg_data_flat, flash_sequence_indices, model_path, features_path):
    """
    Predicts a character from raw EEG data using a pre-trained P300 model.
    """
    # --- Parameters (matching the training notebook and MATLAB script) ---
    SRATE = 600
    NUM_CHANNELS = 3
    NUM_SEQUENCES = 12
    TOTAL_FLASHES = 150
    PRE_STREAM_MS = 200
    FLASH_DURATION_MS = 100
    ISI_DURATION_MS = 50  # Inter-stimulus interval in ms
    
    BASELINE_MS = [-200, 0]
    FRAME_MS = [-200, 1000]
    DOWNSAMPLE_FACTOR = 24

    # --- Load Model and Features ---
    try:
        model = joblib.load(model_path)
        feature_indices = joblib.load(features_path)
    except Exception as e:
        return f"Error loading model/features: {e}"

    # --- Reshape and Preprocess EEG Data ---
    try:
        eeg_data = np.array(eeg_data_flat).reshape(NUM_CHANNELS, -1)
    except Exception as e:
        return f"Error reshaping data: {e}"

    eeg_filtered = butter_bandpass_filter(eeg_data, 0.5, 10, SRATE, 4)

    # --- Generate Event Markers (Corrected Timing) ---
    flash_onsets_samples = []
    pre_stream_samples = int(PRE_STREAM_MS / 1000 * SRATE)
    flash_duration_samples = int(FLASH_DURATION_MS / 1000 * SRATE)
    isi_duration_samples = int(ISI_DURATION_MS / 1000 * SRATE)
    
    # The total time from the start of one flash to the start of the next
    total_event_duration_samples = flash_duration_samples + isi_duration_samples

    # The first flash happens after the pre-stream AND the first ISI
    first_flash_onset = pre_stream_samples + isi_duration_samples

    for i in range(TOTAL_FLASHES):
        # The onset of each subsequent flash is offset from the first flash
        onset = first_flash_onset + (i * total_event_duration_samples)
        flash_onsets_samples.append(onset)
    flash_onsets_samples = np.array(flash_onsets_samples)

    # --- Feature Extraction ---
    all_epochs = extractEpoch3D(eeg_filtered, flash_onsets_samples, SRATE, BASELINE_MS, FRAME_MS, False)
    
    if all_epochs.shape[2] == 0:
        return "Error: No valid epochs extracted."

    down_epochs = decimation_by_avg(all_epochs, DOWNSAMPLE_FACTOR)
    n_ch, n_frames, n_trials = down_epochs.shape
    features = np.reshape(down_epochs, (n_ch * n_frames, n_trials)).transpose()
    features_selected = features[:, feature_indices]

    # --- Prediction ---
    scores = model.predict(features_selected).flatten()

    # --- Determine Target Row and Column ---
    sequence_scores = np.zeros(NUM_SEQUENCES)
    flash_indices_np = np.array(flash_sequence_indices)

    for i in range(NUM_SEQUENCES):
        sequence_mask = (flash_indices_np == i)
        if np.any(sequence_mask):
            sequence_scores[i] = np.mean(scores[sequence_mask])

    row_scores = sequence_scores[:6]
    col_scores = sequence_scores[6:]
    target_row_idx = np.argmax(row_scores)
    target_col_idx = np.argmax(col_scores)

    # --- Map to Character ---
    speller_grid = create_speller_grid()
    predicted_char = speller_grid[target_row_idx, target_col_idx]

    return predicted_char
