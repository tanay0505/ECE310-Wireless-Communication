import os
import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Path to data (use raw string or double backslashes)
data_directory = "C:/Users/shrut/OneDrive/Desktop/Engineering/Mechanical Engineering/Sem 6/Wireless Communications/BVP"
sample_size = 20  # Data dimensions expected
fixed_time_steps = 50  # Standardized time steps after padding or truncation

# Padding or truncation function
def pad_or_truncate(data, target_length):
    if data.shape[2] > target_length:
        # Truncate
        return data[:, :, :target_length]
    elif data.shape[2] < target_length:
        # Pad with zeros
        pad_width = target_length - data.shape[2]
        return np.pad(data, ((0, 0), (0, 0), (0, pad_width)), 'constant')
    else:
        return data

# Function to determine the number of classes
def determine_num_classes(data_directory):
    labels = set()  # Use a set to avoid duplicates
    
    for root, dirs, files in os.walk(data_directory):
        for dir_name in dirs:
            if "6-link" in dir_name:
                user_folder_path = os.path.join(root, dir_name)
                
                for user_folder in os.listdir(user_folder_path):
                    user_path = os.path.join(user_folder_path, user_folder)
                    
                    if os.path.isdir(user_path) and user_folder.startswith("user"):
                        for file in os.listdir(user_path):
                            if file.endswith('.mat'):
                                try:
                                    label = int(file.split('-')[1])  # Extract label from filename
                                    labels.add(label)  # Add label to the set
                                except Exception as e:
                                    print(f"Error extracting label from file {file}: {e}")
    
    return len(labels)  # Number of unique labels

# Determine number of classes
num_classes = 11  # Set to 11 to include labels from 0 to 10
print(f"Number of classes: {num_classes}")

# Load data for training and testing
def load_data(data_directory):
    all_data = []
    labels = []
    found_labels = set()  # Track all unique labels found

    for root, dirs, files in os.walk(data_directory):
        for dir_name in dirs:
            if "6-link" in dir_name:
                user_folder_path = os.path.join(root, dir_name)

                for user_folder in os.listdir(user_folder_path):
                    user_path = os.path.join(user_folder_path, user_folder)

                    if os.path.isdir(user_path) and user_folder.startswith("user"):
                        for file in os.listdir(user_path):
                            if file.endswith('.mat'):
                                file_path = os.path.join(user_path, file)
                                # Check if the file is empty
                                if os.path.getsize(file_path) == 0:
                                    print(f"Skipping empty file: {file_path}")
                                    continue  # Skip empty files
                                
                                try:
                                    mat_data = scipy.io.loadmat(file_path)
                                    if 'velocity_spectrum_ro' in mat_data:
                                        bvp_data = mat_data['velocity_spectrum_ro']
                                        if bvp_data.shape[0:2] == (sample_size, sample_size):
                                            bvp_data = pad_or_truncate(bvp_data, fixed_time_steps)
                                            label = int(file.split('-')[1])
                                            
                                            # Track found labels
                                            found_labels.add(label)

                                            # Ensure label is within valid range
                                            if label < num_classes:  # Check if label is valid
                                                all_data.append(bvp_data)
                                                labels.append(label)
                                            else:
                                                print(f"Skipping invalid label {label} from file: {file_path}")
                                    else:
                                        print(f"'velocity_spectrum_ro' not found in file: {file_path}")
                                except Exception as e:
                                    print(f"Error loading file {file_path}: {e}")

    print(f"Unique labels found in dataset: {found_labels}")  # Print all found labels
    return np.array(all_data), np.array(labels)

# Load BVP data
X, y = load_data(data_directory)
X = X.astype('float32') / 2.0  # Normalize the data

# Ensure y contains only valid labels before one-hot encoding
if np.max(y) >= num_classes:  # Check for any out-of-bounds labels
    raise ValueError(f"Labels contain out-of-bounds values. Max label: {np.max(y)}, Expected max: {num_classes - 1}")

y = to_categorical(y, num_classes)  # One-hot encode labels

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data generator for memory-efficient batch processing
def bvp_data_generator(X, y, batch_size):
    while True:
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]
            yield X_batch, y_batch

# Model architecture with Dropout for regularization
model = Sequential([
    Flatten(input_shape=(sample_size, sample_size, fixed_time_steps)),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Add dropout layer
    Dense(64, activation='relu'),
    Dropout(0.5),  # Add dropout layer
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train model using the generator
batch_size = 100
steps_per_epoch = len(X_train) // batch_size

# Tracking history to plot accuracy and loss
# Train model using the generator (without EarlyStopping)
history = model.fit(
    bvp_data_generator(X_train, y_train, batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=100,  # Set to 100 epochs
    validation_data=(X_test, y_test)
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy*100:.2f}%')

# Plotting accuracy and loss for both training and validation
epochs = range(1, len(history.history['accuracy']) + 1)

plt.figure(figsize=(10, 6))

#Graph
epochs = range(1, len(history.history['accuracy']) + 1)

fig, ax1 = plt.subplots()

# Plot accuracy on the left y-axis
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy', color='tab:red')
ax1.plot(epochs, history.history['accuracy'], 'r-', label='Training Accuracy')
ax1.plot(epochs, history.history['val_accuracy'], 'y-', label='Validation Accuracy')
ax1.tick_params(axis='y', labelcolor='tab:red')

# Create a twin y-axis to plot the loss
ax2 = ax1.twinx()
ax2.set_ylabel('Loss', color='tab:blue')
ax2.plot(epochs, history.history['loss'], 'b-', label='Training Loss')
ax2.plot(epochs, history.history['val_loss'], 'g-', label='Validation Loss')
ax2.tick_params(axis='y', labelcolor='tab:blue')

# Adding legends for both accuracy and loss
fig.tight_layout()
fig.legend(loc='upper right', bbox_to_anchor=(1, 0.9), bbox_transform=ax1.transAxes)

plt.title('Training and Validation Accuracy and Loss')
plt.show()


# Make predictions on the test set(True label Mattrix)
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Plot confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=False, cmap='Blues', fmt='g')

# Add title and labels
plt.title(f'Acc={accuracy:.3f}')
plt.xlabel('True Labels')
plt.ylabel('Predicted Labels')

plt.show()
