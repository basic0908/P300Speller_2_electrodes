import mat73
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import mne
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import warnings
import os

# --- Configuration Parameters ---
# IMPORTANT: Adjust this path to your actual .mat file location
DATA_FILE_PATH = "C:/Users/ryoii/OneDrive/Documents/GitHub/P300Speller_2_electrodes/Python/data/s48.mat"
RESULT_FOLDER = "result" # Folder to save all results

# P300 Speller & Signal Processing Parameters
BASELINE_MS = [-200, 0]  # in ms
EPOCH_FRAME_MS = [0, 600]  # in ms
BANDPASS_FREQ = [0.5, 10] # Hz
FILTER_ORDER = 4
DECIMATION_FACTOR = 24
P_VALUE_THRESHOLD = 0.08
NUM_SELECTED_FEATURES = 60 # Number of features to select after p-value elimination

# P300 Speller Matrix Definition
SPELLER_MATRIX = ['A', 'B', 'C', 'D', 'E', 'F',
                  'G', 'H', 'I', 'J', 'K', 'L',
                  'M', 'N', 'O', 'P', 'Q', 'R',
                  'S', 'T', 'U', 'V', 'W', 'X',
                  'Y', 'Z', '1', '2', '3', '4',
                  '5', '6', '7', '8', '9', '_']

P3_SPELLER_CONFIG = {
    "seq_code": list(range(1, 13)), # 12 flashes (6 rows, 6 columns)
    "full_repeat": 15,             # Number of full repetitions
    "spellermatrix": SPELLER_MATRIX
}

# Suppress DeprecationWarning from numpy.chararray
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- Signal Processing Functions ---

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    """Applies a zero-phase band-pass Butterworth filter with de-meaning."""
    nyq = fs / 2
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    # Demean before filtering
    meandat = np.mean(data, axis=1)
    data_demeaned = data - meandat[:, np.newaxis]
    y = filtfilt(b, a, data_demeaned)
    return y

def extractEpoch3D(data, event_indices, srate, baseline_ms, frame_ms, opt_keep_baseline):
    """
    Extracts 3D epochs [channels x time x trials] from 2D data [channels x time].
    Applies baseline correction.
    """
    baseline_start_s = baseline_ms[0] / 1000
    frame_start_s = frame_ms[0] / 1000
    frame_end_s = frame_ms[1] / 1000

    if opt_keep_baseline:
        begin_offset_samples = int(np.floor(baseline_start_s * srate))
        end_offset_samples = int(np.floor((frame_end_s - baseline_start_s) * srate))
    else:
        begin_offset_samples = int(np.floor(frame_start_s * srate))
        end_offset_samples = int(np.floor((frame_end_s - frame_start_s) * srate))

    epoch_length_samples = end_offset_samples

    epoch3D = np.zeros((data.shape[0], epoch_length_samples, len(event_indices)))
    
    for nth_event, event_i in enumerate(event_indices):
        if opt_keep_baseline:
            begin_id = int(event_i + np.floor(baseline_start_s * srate))
            end_id = int(begin_id + np.floor((frame_end_s - baseline_start_s) * srate))
        else:
            begin_id = int(event_i + np.floor(frame_start_s * srate))
            end_id = int(begin_id + np.floor((frame_end_s - frame_start_s) * srate))
        
        tmp_data = data[:, begin_id:end_id]

        baseline_duration_s = (baseline_ms[1] - baseline_ms[0]) / 1000
        begin_base_samples = int(np.floor(baseline_start_s * srate))
        end_base_samples = int(begin_base_samples + np.floor(baseline_duration_s * srate))
        
        # Ensure baseline indexing is within tmp_data bounds
        if end_base_samples > tmp_data.shape[1]:
            end_base_samples = tmp_data.shape[1] # Adjust if epoch is shorter than full baseline
        
        base = np.mean(tmp_data[:, begin_base_samples:end_base_samples], axis=1)

        rmbase_data = tmp_data - base[:, np.newaxis]
        epoch3D[:, :, nth_event] = rmbase_data

    return epoch3D

def decimation_by_avg(data, factor):
    """
    Replaces sequences of `factor` samples with their average,
    effectively downsampling the data by averaging.
    """
    n_ch, n_frame, n_trial = data.shape
    decimated_frame = int(np.floor(n_frame / factor))
    decimated_data = np.zeros((n_ch, decimated_frame, n_trial))

    for i in range(n_trial):
        for j in range(decimated_frame):
            cur_data_trial = data[:, :, i]
            decimated_data[:, j, i] = np.mean(cur_data_trial[:, j * factor:(j + 1) * factor], axis=1)
    return decimated_data

# --- Statistical and Classification Functions ---

def get_ols_stats(data_matrix, x_columns, y_labels):
    """Performs Ordinary Least Squares regression and returns results."""
    x = data_matrix[:, x_columns]
    results = sm.OLS(y_labels, x).fit()
    return results

def detect_letter_P3speller(pred_scores, word_length, true_label, letter_event_indices, seq_markers, params):
    """
    Detects letters from P300 speller predictions based on specified parameters
    and calculates accuracy per repetition.
    """
    user_answer_chars = np.chararray(word_length, 1)
    acc_on_repetition = np.zeros(params["full_repeat"])
    correct_on_repetition = np.zeros(params["full_repeat"])

    num_seq_codes = len(params["seq_code"])

    for n_repeat in range(params["full_repeat"]):
        for n_letter in range(word_length):
            # Calculate start and end trials for a single letter session
            begin_trial_idx = num_seq_codes * params["full_repeat"] * n_letter
            end_trial_idx = begin_trial_idx + (n_repeat + 1) * num_seq_codes

            speller_code_scores = np.zeros(num_seq_codes)
            for j in range(begin_trial_idx, end_trial_idx):
                # Accumulate scores for each flash sequence (row/column)
                code_index = int(seq_markers[letter_event_indices[j]]) - 1
                speller_code_scores[code_index] += pred_scores[j]

            # Determine the predicted row and column based on highest scores
            row_idx = np.argmax(speller_code_scores[0:6])
            col_idx = np.argmax(speller_code_scores[6:12])
            
            # Map to the character in the speller matrix
            user_answer_chars[n_letter] = params['spellermatrix'][row_idx * 6 + col_idx]
            
        user_answer_string = user_answer_chars.tobytes().decode('utf-8')
        
        # Calculate accuracy for the current number of repetitions
        correct_count = sum(1 for i, j in zip(user_answer_string, true_label) if i == j)
        correct_on_repetition[n_repeat] = correct_count
        acc_on_repetition[n_repeat] = correct_count / len(true_label)

    out = {
        "text_result": user_answer_string,
        "acc_on_repetition": acc_on_repetition,
        "correct_on_repetition": correct_on_repetition
    }
    return out

# --- Main Script Execution for T7 & T8 ---

# Create the Result folder if it doesn't exist
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load Data once
EEG_data = mat73.loadmat(DATA_FILE_PATH)
biosemi_montage = mne.channels.make_standard_montage('biosemi32')
all_ch_names = biosemi_montage.ch_names

# --- Specific Channel Selection: T7 and T8 ---
try:
    t7_idx = all_ch_names.index('T7')
    t8_idx = all_ch_names.index('T8')
    selected_channel_indices = [t7_idx, t8_idx]
    selected_ch_names = ['T7', 'T8']
    num_channels = len(selected_channel_indices) # This will be 2
    analysis_name = "T7_T8"
except ValueError as e:
    print(f"Error: Could not find channel 'T7' or 'T8' in the montage. Please check channel names. Error: {e}")
    exit() # Exit if channels not found

print(f"--- Processing with {num_channels} Channels (T7 and T8) ---")
    
# --- Pre-processing for Training Data ---
all_target_epochs = []
all_nontarget_epochs = []

for n_calib_session in range(len(EEG_data['train'])):
    session_data = EEG_data['train'][n_calib_session]
    # Select only the current subset of channels
    data_subset = np.asarray(session_data['data'])[selected_channel_indices, :]
    srate = session_data['srate']
    
    filtered_data = butter_bandpass_filter(data_subset, BANDPASS_FREQ[0], BANDPASS_FREQ[1], srate, FILTER_ORDER)
    
    markers = session_data['markers_target']
    target_indices = np.where(markers == 1)[0]
    nontarget_indices = np.where(markers == 2)[0]

    tmp_target_eeg = extractEpoch3D(filtered_data, target_indices, srate, BASELINE_MS, EPOCH_FRAME_MS, False)
    tmp_nontarget_eeg = extractEpoch3D(filtered_data, nontarget_indices, srate, BASELINE_MS, EPOCH_FRAME_MS, False)
    
    all_target_epochs.append(tmp_target_eeg)
    all_nontarget_epochs.append(tmp_nontarget_eeg)

target_epochs = np.dstack(all_target_epochs)
nontarget_epochs = np.dstack(all_nontarget_epochs)

# --- Draw Event-Related Potentials (ERPs) ---
avg_target_erp = np.mean(target_epochs, axis=2)
avg_nontarget_erp = np.mean(nontarget_epochs, axis=2)

time_points_ms = np.linspace(EPOCH_FRAME_MS[0], EPOCH_FRAME_MS[1], avg_target_erp.shape[1])

fig_erp = plt.figure(figsize=(8, 5))
# Plot the first channel in the subset (T7) for ERP
plot_channel_idx_in_subset = 0 
plt.plot(time_points_ms, avg_target_erp[plot_channel_idx_in_subset, :].T, color='orange', label='Target')
plt.plot(time_points_ms, avg_nontarget_erp[plot_channel_idx_in_subset, :].T, color='black', label='Non-Target')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude ($\mu V$)')
plt.title(f'Averaged ERPs ({selected_ch_names[plot_channel_idx_in_subset]}) - {num_channels} Channels')
plt.gca().yaxis.grid(True)
plt.rcParams.update({'font.size': 13})
plt.xlim(EPOCH_FRAME_MS)
plt.legend()
plt.savefig(os.path.join(RESULT_FOLDER, f'erp_channels_{analysis_name}.png'))
plt.close(fig_erp) # Close the figure to prevent display

# --- Feature Extraction and Preparation for Classification ---
downsampled_target = decimation_by_avg(target_epochs, DECIMATION_FACTOR)
downsampled_nontarget = decimation_by_avg(nontarget_epochs, DECIMATION_FACTOR)

feat_target_reshaped = downsampled_target.reshape(downsampled_target.shape[0] * downsampled_target.shape[1], 
                                                  downsampled_target.shape[2]).T
feat_nontarget_reshaped = downsampled_nontarget.reshape(downsampled_nontarget.shape[0] * downsampled_nontarget.shape[1], 
                                                        downsampled_nontarget.shape[2]).T

y_target_labels = np.ones((feat_target_reshaped.shape[0], 1))
y_nontarget_labels = -np.ones((feat_nontarget_reshaped.shape[0], 1))

all_features = np.vstack((feat_target_reshaped, feat_nontarget_reshaped))
all_labels = np.vstack((y_target_labels, y_nontarget_labels))

idx_shuffle = np.arange(all_features.shape[0])
np.random.shuffle(idx_shuffle) # Random shuffling as in original code
shuffled_features = all_features[idx_shuffle, :]
shuffled_labels = all_labels[idx_shuffle, :]

# --- Train Linear Classifier ---
selected_feature_cols = np.arange(shuffled_features.shape[1])
while True:
    ols_results = get_ols_stats(shuffled_features, selected_feature_cols, shuffled_labels)
    max_p_value = np.max(ols_results.pvalues)
    
    if max_p_value <= P_VALUE_THRESHOLD:
        break
    else:
        backward_elim_idx = np.array(np.where(ols_results.pvalues == max_p_value))[0]
        selected_feature_cols = np.delete(selected_feature_cols, backward_elim_idx)

sorted_p_values_indices = np.argsort(ols_results.pvalues)
final_selected_features_indices = selected_feature_cols[sorted_p_values_indices[:NUM_SELECTED_FEATURES]]

training_features_selected = shuffled_features[:, final_selected_features_indices]
    
linear_classifier = LinearRegression()
linear_classifier.fit(training_features_selected, shuffled_labels)

predictions_train = np.sign(linear_classifier.predict(training_features_selected))
training_accuracy = np.sum(predictions_train == shuffled_labels) / len(predictions_train)
    
# --- Letter Detection Accuracy on Test Data ---
test_answers_texts = []
test_accuracies = []

for n_test_session in range(len(EEG_data['test'])):
    current_test_eeg = EEG_data['test'][n_test_session]
    # Select only the current subset of channels for test data
    data_test_subset = np.asarray(current_test_eeg['data'])[selected_channel_indices, :]
    srate_test = current_test_eeg['srate']
    
    filtered_test_data = butter_bandpass_filter(data_test_subset, BANDPASS_FREQ[0], BANDPASS_FREQ[1], srate_test, FILTER_ORDER)
    
    word_length_current = int(current_test_eeg['nbTrials'] / (len(P3_SPELLER_CONFIG['seq_code']) * P3_SPELLER_CONFIG['full_repeat']))

    markers_sequence = current_test_eeg['markers_seq']
    flash_event_indices = np.where(np.isin(markers_sequence, P3_SPELLER_CONFIG['seq_code']))[0]

    unknown_epochs = extractEpoch3D(filtered_test_data, flash_event_indices, srate_test, BASELINE_MS, EPOCH_FRAME_MS, False)
    downsampled_unknown = decimation_by_avg(unknown_epochs, DECIMATION_FACTOR)
    
    feat_unknown_reshaped = downsampled_unknown.reshape(downsampled_unknown.shape[0] * downsampled_unknown.shape[1], 
                                                        downsampled_unknown.shape[2]).T

    # Use the trained model and selected features for prediction
    prediction_scores_unknown = linear_classifier.predict(feat_unknown_reshaped[:, final_selected_features_indices])

    answers_result = detect_letter_P3speller(
        prediction_scores_unknown,
        word_length_current,
        current_test_eeg['text_to_spell'],
        flash_event_indices,
        markers_sequence,
        P3_SPELLER_CONFIG
    )
    
    test_answers_texts.append(answers_result['text_result'])
    test_accuracies.append(answers_result['acc_on_repetition'][-1])

# --- Save Combined Visualization and Results ---
fig_summary, axes_summary = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [0.6, 0.4]})

# MNE Sensor Plot (top subplot)
# Plot sensor projection only for T7 and T8
fake_info_current = mne.create_info(ch_names=selected_ch_names, sfreq=srate, ch_types='eeg')
fake_evoked_current = mne.EvokedArray(avg_target_erp, fake_info_current)
# Create a temporary montage with only the selected channels for plotting
temp_montage = mne.channels.make_dig_montage(ch_pos={name: biosemi_montage.get_positions()['ch_pos'][name] for name in selected_ch_names})
fake_evoked_current.set_montage(temp_montage) 
    
fake_evoked_current.plot_sensors(axes=axes_summary[0], show_names=True, show=False)
axes_summary[0].set_title(f'Channel Projection ({", ".join(selected_ch_names)})', fontweight='bold')


# Text Summary (bottom subplot)
summary_text = (
    f"Number of Channels: {num_channels} ({', '.join(selected_ch_names)})\n"
    f"Training Accuracy: {training_accuracy:.4f}\n\n"
    "Test Results:\n"
)
for i, (ans_text, acc) in enumerate(zip(test_answers_texts, test_accuracies)):
    summary_text += f"  Session {i+1}: {ans_text} (Accuracy: {acc:.4f})\n"

axes_summary[1].text(0.05, 0.95, summary_text, transform=axes_summary[1].transAxes,
                     fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
axes_summary[1].axis('off') # Hide axes for text plot

plt.tight_layout()
plt.savefig(os.path.join(RESULT_FOLDER, f'summary_channels_{analysis_name}.png'))
plt.close(fig_summary) # Close the figure to prevent display

print(f"Results for {num_channels} channels ({analysis_name}) saved to '{RESULT_FOLDER}/erp_channels_{analysis_name}.png' and '{RESULT_FOLDER}/summary_channels_{analysis_name}.png'")
print(f"Training Accuracy: {training_accuracy:.4f}")
for i, (ans_text, acc) in enumerate(zip(test_answers_texts, test_accuracies)):
    print(f"  Session {i+1} Test Result: {ans_text} (Accuracy: {acc:.4f})")

print("\nAnalysis complete.")