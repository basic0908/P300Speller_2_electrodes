import numpy as np
import joblib
from scipy.signal import butter, filtfilt
import os

# =============================================================================
# Helper Functions (from the notebook)
# =============================================================================

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    """Applies a Butterworth band-pass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    # Demean before filtering to remove DC offset
    meandat = np.mean(data, axis=1, keepdims=True)
    data = data - meandat
    y = filtfilt(b, a, data, axis=1)
    return y

def extractEpoch3D(data, event_samples, srate, baseline_ms, frame_ms):
    """
    Extracts and baseline-corrects time-locked epochs from continuous EEG data.
    """
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

        base_start_idx = int(i_event_sample + np.floor(baseline_ms[0] / 1000 * srate))
        base_end_idx = int(i_event_sample + np.floor(baseline_ms[1] / 1000 * srate))

        if base_start_idx < 0 or base_end_idx >= data.shape[1] or (base_end_idx - base_start_idx) <= 0:
            continue

        baseline_mean = np.mean(data[:, base_start_idx:base_end_idx], axis=1, keepdims=True)
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

    Args:
        eeg_data_flat (list or np.array): A flat list or 1D numpy array
                                          containing the raw EEG data for one character trial.
                                          Expected shape: (num_channels * num_samples,).
        flash_sequence_indices (list or np.array): A list containing the sequence
                                                   of row/column flashes (0-11).
        model_path (str): Path to the saved classifier model (.pkl).
        features_path (str): Path to the saved feature selection indices (.pkl).

    Returns:
        str: The predicted character.
    """
    # --- Parameters ---
    SRATE = 600  # Hz
    NUM_CHANNELS = 3 # LEFT, RIGHT, DIFF
    NUM_SEQUENCES = 12 # 6 rows + 6 columns
    TOTAL_FLASHES = 150

    PRE_STREAM_MS = 200
    FLASH_DURATION_MS = 100
    
    # Epoch extraction parameters from the notebook
    BASELINE_MS = [-200, 0]
    FRAME_MS = [-200, 1000]
    DOWNSAMPLE_FACTOR = 24

    # --- Load Model and Features ---
    try:
        model = joblib.load(model_path)
        feature_indices = joblib.load(features_path)
    except FileNotFoundError as e:
        return f"Error: Model or feature file not found. {e}"
    except Exception as e:
        return f"Error loading model: {e}"

    # --- Reshape and Preprocess EEG Data ---
    try:
        eeg_data = np.array(eeg_data_flat).reshape(NUM_CHANNELS, -1)
    except ValueError as e:
        return f"Error reshaping data. Expected {NUM_CHANNELS} channels. {e}"

    eeg_filtered = butter_bandpass_filter(eeg_data, 0.5, 10, SRATE, 4)

    # --- Generate Event Markers from Flash Timings ---
    flash_onsets_samples = []
    pre_stream_samples = int(PRE_STREAM_MS / 1000 * SRATE)
    flash_duration_samples = int(FLASH_DURATION_MS / 1000 * SRATE)

    for i in range(TOTAL_FLASHES):
        onset = pre_stream_samples + (i * flash_duration_samples)
        flash_onsets_samples.append(onset)
    flash_onsets_samples = np.array(flash_onsets_samples)

    # --- Feature Extraction ---
    all_epochs = extractEpoch3D(eeg_filtered, flash_onsets_samples, SRATE, BASELINE_MS, FRAME_MS)
    
    # Ensure the number of extracted epochs matches the number of flashes
    if all_epochs.shape[2] != len(flash_sequence_indices):
        print(f"Warning: Mismatch between extracted epochs ({all_epochs.shape[2]}) and flash sequence length ({len(flash_sequence_indices)}). Truncating to the smaller size.")
        min_len = min(all_epochs.shape[2], len(flash_sequence_indices))
        all_epochs = all_epochs[:,:,:min_len]
        flash_sequence_indices = flash_sequence_indices[:min_len]

    if all_epochs.shape[2] == 0:
        return "Error: No valid epochs could be extracted."

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

# =============================================================================
# Example Usage (for testing in Python)
# =============================================================================
if __name__ == '__main__':
    # --- Configuration ---
    if not os.path.exists('models'):
        os.makedirs('models')
        
    MODEL_FILE = 'models/p300_classifier_model.pkl'
    FEATURES_FILE = 'models/p300_feature_selection_indices.pkl'

    # --- Create Dummy Model and Features for Demonstration ---
    try:
        from sklearn.linear_model import LinearRegression
        dummy_model = LinearRegression()
        dummy_model.fit(np.random.rand(10, 90), np.random.rand(10, 1))
        joblib.dump(dummy_model, MODEL_FILE)
        dummy_features = np.arange(90)
        joblib.dump(dummy_features, FEATURES_FILE)
        print(f"Dummy model and features created at '{MODEL_FILE}' and '{FEATURES_FILE}'")
    except Exception as e:
        print(f"Could not create dummy model/features: {e}")

    # --- Simulate Raw EEG Data and Flash Sequence ---
    NUM_SAMPLES = 9720
    NUM_CHANNELS_SIM = 3
    simulated_eeg_data_flat = np.random.randn(NUM_SAMPLES * NUM_CHANNELS_SIM).tolist()

    # This would be generated in your MATLAB script
    REPS_PER_SEQUENCE_SIM = 12.5 # to get 150 total flashes
    flash_sequence = np.tile(np.arange(12), 13) # tile 13 times to get more than 150
    np.random.shuffle(flash_sequence)
    simulated_flash_sequence = flash_sequence[:150].tolist()

    # --- Make Prediction ---
    predicted_character = predict_speller_char(
        simulated_eeg_data_flat, 
        simulated_flash_sequence, 
        MODEL_FILE, 
        FEATURES_FILE
    )

    print("\n" + "="*30)
    print(f"Predicted Character: {predicted_character}")
    print("="*30)
