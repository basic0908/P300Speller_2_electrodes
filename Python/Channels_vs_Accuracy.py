import matplotlib.pyplot as plt
import numpy as np
import os

# Create the Result folder if it doesn't exist
RESULT_FOLDER = "Result"
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Data extracted from your provided log
# Format: num_channels: [acc_session1, acc_session2, acc_session3, acc_session4]
accuracies_data = {
    32: [1.0000, 0.8571, 0.8571, 0.8571],
    30: [1.0000, 0.8571, 1.0000, 1.0000],
    28: [1.0000, 1.0000, 1.0000, 1.0000],
    26: [1.0000, 1.0000, 1.0000, 1.0000],
    24: [1.0000, 1.0000, 1.0000, 1.0000],
    22: [1.0000, 1.0000, 0.8571, 1.0000],
    20: [1.0000, 1.0000, 1.0000, 1.0000],
    18: [1.0000, 1.0000, 1.0000, 1.0000],
    16: [1.0000, 1.0000, 1.0000, 1.0000],
    14: [1.0000, 1.0000, 0.8571, 1.0000],
    12: [1.0000, 1.0000, 0.8571, 1.0000],
    10: [0.8571, 0.8571, 0.8571, 1.0000],
    8: [0.8571, 0.8571, 1.0000, 1.0000],
    6: [1.0000, 0.8571, 0.5714, 1.0000],
    4: [0.8571, 0.8571, 0.4286, 0.8571],
    2: [0.8571, 1.0000, 0.7143, 0.7143]
}

# Prepare data for plotting
channels = sorted(accuracies_data.keys(), reverse=True) # Sort channels in decreasing order
average_accuracies = [np.mean(accuracies_data[c]) for c in channels]
std_dev_accuracies = [np.std(accuracies_data[c]) for c in channels] # For error bars

# Plotting
plt.figure(figsize=(10, 6))
plt.errorbar(channels, average_accuracies, yerr=std_dev_accuracies, fmt='-o', capsize=5, label='Average Accuracy $\pm$ Std. Dev.')
plt.plot(channels, average_accuracies, 'ro', markersize=6) # Individual points

plt.title('P300 Speller Test Accuracy vs. Number of Channels')
plt.xlabel('Number of Channels')
plt.ylabel('Average Test Accuracy')
plt.xticks(channels) # Ensure all channel numbers are shown on x-axis
plt.ylim(0.4, 1.05) # Set y-axis limits to better visualize accuracy range
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

# Save the plot
plot_filename = os.path.join(RESULT_FOLDER, 'channels_vs_accuracy_plot.png')
plt.savefig(plot_filename)
plt.close() # Close the plot to prevent it from displaying

print(f"\nPlot 'channels_vs_accuracy_plot.png' saved to the '{RESULT_FOLDER}' folder.")
print("The plot visually represents the average test accuracy for each channel configuration.")