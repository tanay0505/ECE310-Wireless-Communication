import os
import numpy as np
import scipy.io as scio
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, TimeDistributed, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Global variable for T_MAX
T_MAX = 50  # You can adjust this based on your data

# Function to normalize the data
def normalize_data(data):
    """ Normalize the data. Modify this function based on your dataset requirements. """
    normed_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return normed_data

# Function to zero-pad the data
def zero_padding(data, t_max):
    """ Zero-pad the data along the time dimension to match T_MAX. """
    padded_data = []
    for d in data:
        if len(d.shape) == 3 and d.shape[2] < t_max:  # Check if data has the expected shape (20, 20, time_steps)
            padding = np.zeros((d.shape[0], d.shape[1], t_max - d.shape[2]))
            d_padded = np.concatenate((d, padding), axis=2)
        elif len(d.shape) == 3 and d.shape[2] >= t_max:
            d_padded = d[:, :, :t_max]
        else:
            print(f"Unexpected shape {d.shape}, skipping this sample.")
            continue  # Skip any data that doesn't match the expected dimensions
        padded_data.append(d_padded)
    
    return np.array(padded_data)

# Function to load data
def load_data(path_to_data, motion_sel):
    global T_MAX
    data = []
    label = []
    
    for data_root, data_dirs, data_files in os.walk(path_to_data):
        for data_file_name in data_files:
            file_path = os.path.join(data_root, data_file_name)
            try:
                # Load the data from the .mat file
                mat_data = scio.loadmat(file_path)
                
                # Ensure the key 'velocity_spectrum_ro' exists in the .mat file
                if 'velocity_spectrum_ro' not in mat_data:
                    print(f"Key 'velocity_spectrum_ro' not found in {file_path}, skipping.")
                    continue

                data_1 = mat_data['velocity_spectrum_ro']
                label_1 = int(data_file_name.split('-')[1])

                if label_1 not in motion_sel:
                    continue

                # Normalize the data
                data_normed_1 = normalize_data(data_1)

                # Append to the data list
                data.append(data_normed_1)

                # Append corresponding label
                label.append(label_1)

                # Update T_MAX if necessary
                if len(data_1.shape) == 3 and T_MAX < data_1.shape[2]:
                    T_MAX = data_1.shape[2]
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

    # Ensure that we are padding data only if it has valid samples
    if len(data) == 0:
        raise ValueError("No valid data found. Please check your dataset and directory path.")
    
    # Zero-padding the data to ensure uniform time length
    data = zero_padding(data, T_MAX)

    # Swap axes: [N, 20, 20', T_MAX] => [N, T_MAX, 20, 20']
    data = np.swapaxes(np.swapaxes(data, 1, 3), 2, 3)
    data = np.expand_dims(data, axis=-1)  # [N, T_MAX, 20, 20] => [N, T_MAX, 20, 20, 1]

    return np.array(data), np.array(label)

# Path to your data directory
data_dir = "C:/Users/shrut/OneDrive/Desktop/Engineering/Mechanical Engineering/Sem 6/Wireless Communications/BVP"  # Update this path if needed

# List of selected motions (replace with actual motion labels)
ALL_MOTION = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Load data
data, label = load_data(data_dir, ALL_MOTION)

# Define number of unique motions
N_MOTION = len(np.unique(label))

# One-hot encode the labels
lb = LabelBinarizer()
label = np.reshape(label, (-1, N_MOTION))

# Model definition
model = Sequential([
    TimeDistributed(Flatten(), input_shape=(T_MAX, 20, 20, 1)),
    LSTM(128, activation='tanh', return_sequences=False),
    Dropout(0.5),
    Dense(N_MOTION, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Train the model
history = model.fit(data, label, epochs=10, batch_size=32, validation_split=0.2)

# Save the trained model
model.save('lstm_model.h5')

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
    
# Get final training accuracy
final_train_accuracy = history.history['accuracy'][-1]  # or 'acc' for older versions
print(f'Final Training Accuracy: {final_train_accuracy*100:.4f}')

